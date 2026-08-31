/**
 * Artificial -> Cropland: similarity map and sample acquisition
 * =============================================================
 * Earth Engine Code Editor (JavaScript). Paste whole, press Run.
 *
 * WHY THIS SCRIPT EXISTS
 * ----------------------
 * `Artificial -> Cropland` is the dead class. 46 labelled plots; the deployed
 * model's posterior for it never exceeds 0.191 anywhere in 21 M pixels and it
 * wins the arg-max on 1. So it cannot be found with the model -- every
 * model-in-the-loop acquisition score (entropy, margin, BALD, conformal set
 * size) is blind to it by construction. This script is the model-free retrieval
 * channel: it ranks land by how much its AlphaEarth 2018->2024 *change vector*
 * looks like the change vector of the 46 plots we do have.
 *
 * WHAT WAS MEASURED BEFORE WRITING IT (on the 6,490-plot labelled frame)
 * ---------------------------------------------------------------------
 *   representation                     AUC      target plots in top 100
 *   2018 embedding only                0.820     9
 *   2024 embedding only                0.806    11
 *   concat(2018, 2024)                 0.825    13
 *   >> normalised difference vector    0.915    18.4 +- 2.6  (leave-one-out)
 *
 * Base rate is 46/6490 = 0.71%, so 18 in the top 100 is a ~26x enrichment.
 * That is the number this script is worth; it is the same order as the 15x in
 * the bootstrapping paper this design follows (arXiv:2403.02736).
 *
 * Note the difference vector wins by a wide margin. This does NOT reopen the
 * ledger's "normalised-difference features tested negative" -- that verdict is
 * about per-band ND features as *classifier inputs*. This is the L2-normalised
 * difference *vector* used as a *retrieval direction*. Different use.
 *
 * THE ONE THING TO UNDERSTAND BEFORE TRUSTING THE MAP
 * --------------------------------------------------
 * The top 100 by similarity break down as:
 *
 *   Artificial -> Nature      31   <-- the dominant contaminant
 *   Artificial -> Cropland    26   <-- target
 *   Artificial -> Artificial  16
 *   Nature -> Nature          14
 *   everything else           13
 *
 * 73 of 100 start from Artificial, against a 17.8% base rate. So the AlphaEarth
 * channel is really detecting **de-urbanisation** -- it is strong on "was this
 * built-up un-built?" and weak on "un-built into what?".
 *
 * Two consequences, and they shape the whole script:
 *
 *  1. Dynamic World supplies the DESTINATION, which the embedding cannot. The
 *     division of labour is deliberate: AlphaEarth answers *did it de-urbanise*,
 *     Dynamic World answers *did it become cropland rather than nature*. Their
 *     errors are independent, which is the whole point of an auxiliary layer
 *     (see the section-P negative on correlated single-date paths).
 *
 *  2. Send these candidates to interpreters as an **Artificial -> {Cropland,
 *     Nature}** discrimination task, not as a yes/no on Cropland. The 31
 *     contaminants are not waste: `Artificial -> Nature` holds 123 plots and is
 *     itself in deficit at ~700 patches. One campaign, two deficit classes.
 *
 * OUTPUTS
 * -------
 *   Layers      similarity, de-urbanisation, to-cropland, novelty, activation,
 *               the composite acquisition score, and the 46 known plots
 *   Export      top-N 5 km cell centroids as a CSV of candidate patches,
 *               shaped to drop straight into plan_patch_sampling.py
 *
 * Constants come from src/gee/make_art_crop_constants.py -- rerun it if the
 * label frame changes. 4dp rounding was checked: the top-100 set is identical.
 */

// ===========================================================================
// CONFIG
// ===========================================================================
var AOI = ee.Geometry.Rectangle([112.0, 32.0, 120.0, 38.0]);  // N China Plain
// Global run: comment the line above, uncomment below, and expect a long job.
// var AOI = ee.Geometry.Rectangle([-180, -60, 180, 75], null, false);

var YEAR_A = 2018;         // first endpoint, matches the deployed model
var YEAR_B = 2024;         // second endpoint
var CELL_M = 5000;         // labelling unit: 5 km, matches global_patches.py
var TOP_N  = 200;          // candidate cells to export
var SIM_MIN = 0.384;       // 10th percentile of the 46 target plots
var MIN_HA  = 1.0;         // usable ground per cell; PATCH_SAMPLING.md section B
var EXPORT_FOLDER = 'recoverHabloss_al';

// Weights for the composite score. Deliberately not tuned -- there is no
// held-out set to tune them on, and a tuned weight here would be a fabricated
// number in the place the plan is weakest. Equal until the pilot measures them.
var W = {similarity: 1.0, destination: 1.0, novelty: 0.0};


var PROTOTYPES = [
    [-0.1764, 0.1233, -0.0307, 0.0594, -0.0485, -0.1778, -0.0366, 0.1664, 0.3028, -0.2080, -0.0677, 0.0701, -0.1418, 0.0419, 0.1593, 0.0462, -0.0988, 0.0491, -0.1495, 0.0637, -0.0556, 0.1003, 0.0876, 0.0941, 0.1018, -0.0965, 0.1170, -0.2011, 0.0981, -0.0269, -0.2684, 0.0361, -0.2159, 0.2232, -0.0183, 0.0856, 0.1977, 0.1620, 0.0675, -0.1621, 0.0531, 0.1469, -0.1370, -0.0935, -0.0155, -0.0312, 0.1590, -0.1675, 0.1834, 0.1431, 0.1417, 0.0837, -0.1183, 0.1655, -0.0287, 0.1082, -0.0726, -0.0149, -0.0381, -0.0497, -0.1013, -0.0089, 0.0705, 0.0109],
    [-0.1790, 0.0293, 0.0426, -0.0122, -0.1189, 0.1504, -0.1159, 0.1933, 0.2608, -0.0578, 0.0211, 0.0247, 0.0244, 0.2323, 0.2143, 0.1748, -0.2170, -0.1762, -0.0053, -0.2302, -0.0987, -0.1146, 0.0124, 0.0908, -0.0262, -0.1542, 0.0921, -0.0672, -0.0800, 0.0204, -0.0959, 0.0046, -0.0286, 0.0148, -0.1131, 0.2996, 0.1074, -0.0579, -0.0143, 0.2428, 0.0896, 0.2188, 0.0570, -0.1824, -0.1531, -0.1275, 0.1785, -0.0074, -0.0286, -0.1078, 0.0342, 0.0775, -0.0586, -0.1411, 0.0098, 0.1189, -0.0493, 0.0379, -0.1724, 0.0575, -0.1064, -0.0431, 0.0958, 0.0104],
    [0.0073, -0.0261, -0.0929, 0.0241, 0.0735, 0.1107, 0.0409, 0.0801, 0.2462, -0.1431, -0.0148, 0.1631, 0.0115, -0.0390, 0.2772, 0.3341, -0.0937, 0.1432, -0.0397, -0.3133, -0.0707, -0.0406, 0.0213, 0.1229, 0.1915, -0.2050, 0.1492, -0.0010, 0.0522, -0.0492, 0.0804, 0.0516, -0.0403, -0.0482, -0.2507, -0.0350, 0.0027, -0.0555, 0.1805, 0.0908, 0.2064, 0.0273, -0.0157, 0.0438, 0.0902, -0.1752, -0.0839, -0.0920, -0.1979, -0.0528, -0.1382, -0.1605, -0.0502, -0.0400, -0.0791, 0.0156, 0.0309, -0.0568, -0.1434, 0.1101, 0.0578, 0.1993, 0.0822, 0.0624]
  ];
var NOVELTY_CENTROIDS = [
    [-0.1462, 0.1077, 0.0175, -0.0098, 0.0730, -0.1312, 0.0122, -0.2202, -0.0378, -0.0424, -0.0725, -0.1025, -0.1065, -0.1128, -0.1898, 0.1727, 0.0669, 0.0745, -0.2891, 0.1115, -0.1090, 0.1160, -0.0385, 0.0564, -0.1139, -0.0621, 0.1244, -0.1469, 0.1142, 0.0349, -0.1549, -0.0178, 0.0051, 0.0304, -0.0422, 0.1205, 0.1413, 0.2237, -0.1921, -0.0583, 0.2762, 0.1232, -0.0779, -0.0498, 0.1732, 0.0223, 0.1309, -0.1138, -0.0161, -0.1390, -0.0099, 0.2417, 0.0602, 0.0271, -0.0498, -0.0988, -0.0266, -0.2424, 0.2983, -0.0945, -0.0033, -0.0516, -0.0350, -0.1182],
    [0.0073, 0.2333, -0.0448, 0.0983, 0.1359, 0.0302, -0.0568, -0.0177, 0.1592, 0.0455, -0.0374, 0.0762, -0.1091, 0.2006, -0.0941, 0.0534, 0.0538, -0.1028, -0.0478, -0.0274, -0.1522, 0.1593, 0.0320, -0.0135, -0.0128, -0.0038, 0.1846, 0.0089, 0.3057, -0.0074, 0.0202, 0.1764, -0.1762, 0.1316, 0.1505, -0.2444, -0.1178, 0.1989, -0.0631, 0.0709, -0.2286, 0.1699, 0.0280, 0.1791, 0.0991, 0.1771, 0.0416, -0.1263, 0.0868, 0.1326, -0.0994, -0.0524, -0.1704, -0.0742, 0.1118, -0.1639, -0.0354, -0.1058, 0.1725, 0.0732, -0.1891, -0.1450, -0.0647, 0.0243],
    [0.0174, 0.0397, 0.2356, -0.0554, -0.0519, 0.1538, 0.0330, -0.1103, -0.1536, -0.1065, 0.1393, -0.1151, -0.0998, 0.0921, -0.1094, -0.0214, 0.2522, -0.0016, 0.0694, -0.0774, 0.1153, 0.0524, 0.0229, -0.2196, -0.1020, -0.1704, 0.0160, 0.0659, -0.1389, 0.0861, 0.1632, 0.0157, 0.0520, -0.0939, -0.0799, -0.0114, -0.3066, 0.0562, -0.2240, 0.1062, -0.0707, -0.0446, -0.0903, -0.0228, -0.0801, 0.1687, 0.0553, -0.0363, 0.0300, -0.2424, 0.1248, 0.2036, -0.1926, -0.1010, 0.0585, -0.1313, 0.0940, 0.1441, 0.3120, 0.1103, 0.0791, -0.0394, 0.0305, -0.0253],
    [0.0403, 0.0018, -0.1712, -0.1138, 0.2062, 0.1647, 0.0574, -0.3266, -0.1952, 0.0481, -0.2050, -0.0510, -0.2398, 0.0271, -0.0299, 0.2691, 0.0618, 0.0565, 0.0879, -0.0490, -0.0876, -0.0650, -0.0795, 0.0441, 0.0858, -0.1092, 0.0296, 0.0375, 0.1448, -0.0192, 0.0786, -0.0569, 0.1776, -0.0800, 0.0339, 0.0122, -0.0417, -0.0518, -0.1558, 0.2546, 0.0220, -0.0154, 0.0003, 0.0367, 0.2789, -0.0739, -0.0566, 0.0666, 0.0450, 0.0088, -0.1591, -0.1852, 0.1693, 0.0745, -0.0329, -0.0989, -0.1557, -0.1031, 0.2571, 0.1743, -0.0063, -0.0175, 0.0709, -0.0333],
    [0.1644, -0.1020, 0.0350, 0.1800, 0.1143, -0.1817, 0.1104, 0.1212, 0.2098, 0.1334, 0.1116, -0.0050, 0.1951, 0.1172, 0.0420, 0.0412, -0.0733, 0.0934, 0.0159, 0.0235, -0.1016, -0.0052, 0.0478, 0.1941, 0.0117, 0.2319, -0.0626, -0.0637, -0.0909, 0.1441, -0.1826, -0.0014, -0.2329, -0.0221, -0.1678, 0.0203, 0.0569, 0.0968, -0.1193, 0.0853, -0.1025, -0.2060, -0.1934, -0.1061, 0.3321, -0.0563, -0.0013, 0.0829, -0.0046, 0.1802, -0.0247, -0.0444, -0.0954, 0.0980, -0.0679, 0.1535, -0.1492, 0.0736, -0.1288, -0.0876, -0.2452, 0.0334, 0.0165, 0.0473],
    [0.1113, -0.1003, 0.0734, 0.0131, 0.0512, -0.1681, -0.0042, -0.1152, -0.3767, 0.0807, -0.0487, -0.0542, 0.0079, -0.0897, -0.2590, -0.1820, 0.3422, 0.0843, 0.1612, 0.1582, 0.1347, -0.0130, 0.0091, -0.1481, -0.0335, 0.1159, -0.0544, 0.1333, -0.0682, 0.0341, 0.0637, -0.0266, 0.0645, -0.0744, 0.0991, -0.1781, -0.2395, 0.0199, -0.1454, -0.1181, -0.0820, -0.1121, -0.0779, 0.0601, 0.0713, 0.1621, -0.0962, 0.1070, 0.0572, -0.0209, 0.0046, 0.0767, -0.1148, 0.0161, 0.0834, -0.2050, 0.1060, 0.0705, 0.2395, 0.0074, 0.1017, -0.0670, -0.1320, -0.0360],
    [-0.0775, -0.2628, -0.1282, 0.2433, 0.0942, -0.1292, -0.0176, 0.1519, -0.0336, -0.0510, -0.0233, -0.1454, -0.0338, -0.1273, 0.1206, 0.0937, -0.0752, 0.0518, 0.1360, 0.1095, -0.0045, -0.0334, -0.0550, 0.0635, 0.2236, -0.1436, -0.0332, -0.2863, 0.0472, 0.1691, -0.1637, 0.0785, -0.2206, 0.0978, -0.1203, 0.1793, 0.0095, -0.0055, -0.1152, -0.0161, 0.2222, 0.1023, -0.0517, 0.1364, 0.2650, -0.2067, -0.0414, 0.0090, 0.0296, -0.0333, -0.1210, -0.0298, -0.0135, 0.0938, -0.0928, 0.0845, -0.0693, 0.0306, 0.0056, 0.2627, 0.1127, 0.1166, -0.0924, -0.0590],
    [-0.1171, 0.0126, 0.1173, -0.0797, -0.0957, -0.1025, -0.2569, 0.0941, -0.1619, 0.1369, 0.2019, 0.0987, -0.0402, -0.1319, 0.1101, 0.0352, 0.1440, -0.1002, 0.1813, 0.1416, 0.0349, -0.2363, -0.3108, -0.1242, -0.0564, -0.0720, -0.0648, -0.1735, -0.1097, 0.0189, 0.0127, -0.0553, -0.1056, 0.0049, -0.1828, -0.0515, -0.0696, 0.2225, 0.1652, -0.2362, 0.0208, 0.0286, 0.0438, 0.0466, -0.1066, -0.0342, 0.0029, -0.1130, -0.0555, -0.0631, -0.0282, 0.0188, 0.1325, 0.0784, 0.0538, -0.0209, 0.1136, 0.1447, 0.2190, 0.1258, -0.2314, 0.0511, -0.1026, -0.1331],
    [0.0014, -0.0416, 0.0806, -0.1106, 0.0580, 0.1124, -0.0786, -0.3077, -0.2067, 0.0314, 0.1432, -0.0893, 0.0414, -0.0648, -0.1160, -0.2250, 0.1726, 0.2141, -0.0069, 0.0287, -0.0347, -0.0627, -0.0182, -0.0539, -0.1665, 0.1235, 0.0754, 0.1132, -0.0753, 0.0962, 0.1576, -0.1326, 0.1986, -0.0694, 0.0893, -0.0952, -0.1882, 0.0883, 0.0154, 0.1176, -0.1542, -0.2100, 0.1253, -0.0308, 0.0247, 0.1823, -0.0950, 0.0733, 0.0988, -0.0755, -0.0681, 0.1431, 0.0699, 0.0075, 0.1796, -0.2658, 0.0290, 0.0719, 0.2399, -0.0950, 0.0838, -0.1292, -0.0369, -0.1391],
    [0.1236, -0.0793, 0.1728, 0.0382, 0.1187, 0.0841, -0.0410, -0.1707, -0.1724, 0.0196, 0.0540, 0.0920, 0.0221, -0.0617, -0.0671, 0.0602, 0.1987, -0.1015, -0.0633, -0.2357, 0.1583, 0.0543, -0.0106, 0.2055, -0.0807, 0.0661, -0.0483, 0.3647, 0.0640, -0.1204, 0.0828, 0.0319, 0.2618, 0.0895, -0.2838, 0.0830, -0.0831, -0.0712, -0.0060, -0.1035, 0.1152, 0.0505, 0.0056, -0.0045, 0.1996, -0.1194, -0.0517, -0.0263, -0.2404, -0.0981, -0.1240, -0.0782, -0.1243, -0.0409, 0.0908, -0.1930, -0.0501, 0.1035, -0.0985, 0.1502, 0.0893, 0.0230, 0.1449, 0.0484],
    [-0.0369, 0.0984, 0.2063, -0.0638, -0.0421, -0.1343, -0.0372, -0.1050, -0.0390, -0.0699, -0.1691, 0.1884, -0.0050, -0.0778, -0.0495, 0.0198, 0.0716, 0.2787, 0.0835, 0.0355, 0.0739, 0.2477, 0.0924, 0.2142, -0.0126, -0.0766, -0.1663, 0.0110, 0.1873, -0.0554, 0.0725, -0.0809, 0.1782, 0.0054, 0.0922, -0.1243, -0.0492, 0.0829, -0.0716, -0.3742, -0.2475, -0.1109, -0.1366, -0.0308, 0.1313, 0.1031, -0.0085, 0.1164, 0.1211, 0.2478, -0.0592, 0.0070, -0.1149, 0.0184, 0.0262, -0.1040, -0.0078, -0.1875, 0.1082, -0.0801, -0.1272, -0.1096, 0.0380, -0.0480],
    [0.1920, -0.0693, 0.0018, 0.0423, 0.0501, -0.1907, 0.0814, -0.1625, -0.1286, -0.1281, -0.1259, -0.0408, -0.1014, -0.1175, -0.1514, -0.2459, 0.1735, -0.0501, 0.0480, 0.2337, 0.0101, -0.0001, -0.0929, -0.0549, 0.0383, 0.1995, 0.0201, -0.0195, 0.0690, 0.0176, 0.0548, -0.0262, -0.1020, 0.1244, 0.1674, -0.1145, -0.1777, -0.0304, -0.0368, -0.2613, -0.1406, 0.0013, -0.0339, 0.1359, 0.1012, 0.0154, -0.1965, -0.0710, -0.1212, 0.1440, -0.1562, -0.1112, -0.1541, 0.1109, 0.0802, -0.1563, 0.0422, -0.0069, 0.0205, 0.0419, 0.1269, -0.0378, -0.3047, 0.2418],
    [-0.0342, -0.0458, 0.0641, 0.0882, -0.1793, -0.1344, -0.1749, 0.0439, -0.0491, -0.0906, 0.1757, -0.1486, -0.0354, -0.0743, -0.0310, -0.1539, -0.0346, 0.1015, 0.1176, 0.3558, -0.1147, 0.2080, 0.0421, -0.2186, -0.1577, 0.2533, 0.0832, -0.1954, -0.0750, 0.0596, -0.1569, 0.0447, -0.1399, 0.1163, -0.0482, 0.0679, 0.0264, 0.0881, -0.1151, -0.1758, -0.0718, 0.0618, 0.0570, -0.0153, -0.1220, 0.1405, 0.0394, -0.0053, 0.1141, -0.0186, 0.0769, 0.1011, 0.1649, 0.2159, -0.0937, -0.0321, -0.0103, 0.1096, 0.1824, -0.0856, 0.1703, -0.0881, -0.0655, -0.1762],
    [-0.1262, 0.2117, 0.2386, 0.0645, -0.0791, 0.3089, 0.1318, -0.1495, 0.0724, -0.0977, 0.1349, -0.0870, -0.0596, 0.1103, 0.0537, 0.0656, 0.1011, 0.2065, -0.0652, -0.1246, -0.0583, 0.2217, -0.3165, -0.0151, -0.0598, -0.1064, 0.0012, -0.0699, 0.0005, -0.0725, 0.0035, 0.1248, -0.0406, 0.1403, -0.0519, 0.2424, -0.0839, 0.1097, 0.0086, 0.1332, -0.0145, -0.0370, -0.1144, 0.0048, -0.0985, 0.1591, 0.2308, -0.0761, -0.1172, -0.0416, 0.0936, 0.1504, -0.0899, 0.0126, 0.1651, -0.0362, 0.0241, 0.0720, 0.0380, -0.1367, -0.0529, 0.2076, 0.1437, -0.0504],
    [-0.1259, 0.0404, -0.0735, 0.1538, -0.1307, -0.2333, -0.0409, 0.2921, 0.3203, -0.1883, -0.0576, 0.1309, 0.0000, 0.0801, 0.1886, 0.0840, -0.2159, -0.1251, -0.0678, 0.0336, -0.0138, 0.1126, 0.0350, 0.1267, 0.1038, -0.1330, 0.0497, -0.1761, 0.0637, 0.0408, -0.2177, 0.0097, -0.2666, 0.1449, 0.0143, 0.1582, 0.1919, -0.0056, 0.0302, -0.0861, 0.0078, 0.1871, -0.0029, 0.0030, -0.0210, -0.0879, 0.1306, -0.0336, 0.0744, 0.1199, -0.0567, 0.0102, -0.0848, 0.0831, -0.1074, 0.1422, 0.0192, -0.0865, -0.2285, -0.0863, -0.0684, 0.0155, -0.0165, 0.0917],
    [0.1714, 0.0440, 0.1029, 0.2294, 0.0240, 0.0600, -0.0370, -0.0991, -0.0029, -0.0884, 0.0414, 0.1191, 0.1627, -0.0355, -0.0620, 0.0898, 0.2521, 0.0685, 0.0193, 0.0556, -0.1044, 0.0613, -0.0679, 0.0151, -0.1951, 0.0879, 0.2034, 0.0529, 0.1919, 0.1991, 0.0516, -0.2447, 0.0120, -0.0099, -0.1336, -0.0761, -0.1544, -0.0945, -0.1334, 0.1095, -0.0218, 0.2736, 0.0188, -0.0331, -0.0772, 0.0778, -0.2038, -0.1449, -0.0649, -0.1170, -0.3113, -0.1561, -0.1923, 0.1236, -0.0407, -0.0707, 0.1549, -0.0473, 0.0039, -0.0182, -0.0423, -0.0479, 0.0101, 0.1997],
    [-0.0379, -0.0861, 0.0663, 0.0094, -0.0227, 0.1143, -0.1681, 0.0466, 0.2365, -0.1112, 0.2465, 0.0552, -0.0240, 0.1397, 0.1974, 0.2522, -0.1018, -0.1571, -0.0546, -0.2447, -0.0461, -0.0301, 0.0062, -0.0925, -0.1587, -0.1022, 0.1116, -0.0436, -0.0451, 0.1721, -0.0088, -0.0233, -0.1066, -0.0501, -0.2991, 0.1535, 0.0717, 0.0492, 0.0228, 0.3547, 0.1268, 0.1404, 0.1169, -0.0521, -0.1273, -0.1048, 0.1079, -0.0458, -0.0063, -0.2691, 0.0495, 0.0083, -0.0183, -0.0332, -0.1340, 0.0699, 0.0008, -0.0057, 0.0118, 0.1265, 0.0301, 0.1030, 0.1233, 0.0036],
    [-0.1202, -0.1043, 0.0746, 0.0219, -0.0463, -0.1232, -0.0380, -0.0870, 0.0758, -0.1175, 0.2188, 0.1313, -0.0273, -0.0531, -0.0115, -0.0809, 0.1021, -0.2609, -0.0494, -0.1006, 0.2895, 0.1610, -0.1373, 0.0693, -0.1351, -0.0941, -0.1670, 0.1175, 0.0888, 0.0411, 0.1413, -0.1926, 0.1552, 0.0435, -0.2210, 0.2355, -0.1725, 0.0909, 0.0609, -0.1581, -0.1151, -0.0657, 0.1795, 0.0017, -0.2049, -0.1286, 0.1556, 0.0998, -0.0635, -0.1667, 0.0208, 0.0733, 0.0108, 0.1588, 0.0101, -0.1066, -0.2137, -0.0060, 0.0941, 0.0696, 0.0529, -0.0476, 0.0386, 0.0881],
    [0.1044, 0.0136, 0.2618, -0.0255, -0.1124, 0.1057, -0.0324, 0.0969, 0.0202, -0.1296, -0.0504, 0.0672, -0.0907, -0.0736, 0.1325, 0.1955, 0.0766, 0.1634, 0.2283, -0.1331, -0.2944, 0.0109, 0.2732, -0.0643, -0.0426, -0.0890, 0.1096, 0.0303, -0.1459, -0.0545, 0.0243, 0.0225, 0.1947, 0.0511, 0.0107, -0.0528, 0.1181, -0.1183, -0.1835, -0.0048, 0.0077, 0.2080, -0.1716, 0.0336, 0.1588, 0.0597, -0.2671, -0.1134, 0.0125, 0.1583, -0.0843, 0.0085, -0.2129, -0.0527, -0.0232, -0.0590, 0.1984, 0.0074, 0.0529, 0.0424, 0.0765, -0.0708, -0.1753, 0.1172],
    [-0.1403, 0.1145, 0.0145, -0.0332, 0.0409, -0.0748, -0.1534, 0.4009, 0.0963, -0.0772, 0.1044, -0.0229, 0.0080, 0.0574, -0.0165, -0.1364, -0.0138, -0.1208, 0.2260, 0.0047, 0.0134, -0.1272, 0.0816, -0.1147, 0.1186, -0.1304, -0.0471, -0.1712, -0.0545, 0.2918, 0.0175, -0.1983, -0.0640, 0.1791, 0.1549, 0.1168, 0.0907, 0.1495, -0.0029, 0.1042, -0.1262, -0.0980, 0.0114, 0.1789, -0.0728, 0.1012, 0.1375, 0.0345, 0.1396, 0.0766, 0.0988, 0.1965, -0.1362, -0.0255, 0.0227, -0.0771, 0.0664, -0.0287, 0.2417, -0.1704, -0.0958, 0.0870, -0.1253, 0.0093],
    [-0.0599, 0.0161, -0.0744, -0.0080, 0.0010, 0.0659, -0.0300, 0.0700, 0.0437, -0.0415, 0.2322, -0.2287, 0.2245, -0.0794, 0.2002, -0.2766, -0.1141, 0.1262, 0.1436, -0.0093, -0.2040, -0.0577, -0.0242, 0.1684, 0.1802, -0.1435, 0.0951, -0.1552, -0.0619, 0.2696, -0.0072, -0.1484, 0.0199, 0.0599, -0.0748, 0.1061, -0.2504, -0.0169, 0.1951, 0.0811, 0.0495, -0.0825, 0.2393, -0.0978, -0.0016, -0.0850, 0.0683, 0.1227, 0.1679, -0.1015, -0.1240, 0.0617, 0.0642, -0.0630, -0.0084, -0.0940, 0.0127, 0.0941, -0.1290, -0.0480, 0.1385, -0.0946, -0.0867, -0.1382],
    [0.0541, -0.2919, -0.0286, 0.0438, -0.0496, -0.2616, -0.1119, 0.1936, -0.0193, -0.1380, -0.0536, 0.1317, -0.0518, -0.0164, -0.0751, -0.0451, 0.1350, -0.0982, 0.0709, 0.0204, 0.0466, -0.0122, 0.4495, -0.0868, -0.0326, -0.1672, 0.0738, 0.0813, 0.0399, 0.1679, 0.0765, -0.1376, 0.1001, -0.1040, -0.0772, -0.2769, -0.0222, -0.0477, 0.1231, -0.1496, 0.0273, -0.0924, 0.0246, 0.1725, 0.1732, -0.0510, -0.0656, 0.0446, 0.1129, 0.0060, -0.0388, -0.0356, -0.1390, -0.0526, -0.1650, -0.1218, 0.0274, -0.0430, -0.1131, 0.1530, 0.1045, -0.0824, -0.1684, 0.1009],
    [-0.0130, -0.1381, -0.1058, 0.1116, 0.0905, -0.0192, -0.0579, 0.1669, -0.2181, 0.0627, -0.1901, 0.1067, 0.1326, 0.0422, -0.0459, 0.0585, 0.0144, 0.1681, -0.0408, 0.1113, -0.1846, -0.1220, 0.0877, 0.0210, 0.1801, 0.3404, -0.0404, 0.0281, -0.0290, 0.0652, -0.2810, -0.0355, -0.2156, -0.0516, 0.2717, -0.0301, 0.1084, -0.1584, -0.1085, 0.1327, 0.0458, -0.1462, -0.0756, 0.1543, 0.2235, 0.1890, 0.0113, 0.1297, -0.0842, 0.0617, -0.0669, 0.0595, 0.1787, 0.0314, -0.0875, 0.0265, 0.1205, 0.0166, -0.0852, -0.1308, 0.0492, 0.0861, -0.0006, 0.0528],
    [-0.0584, 0.1485, -0.0060, -0.1741, -0.0762, -0.0340, -0.0132, -0.0711, 0.0339, -0.0123, 0.0535, 0.1495, 0.0915, 0.1472, 0.0315, -0.1459, -0.0377, 0.0163, -0.0022, -0.1767, -0.1431, -0.0225, -0.0170, 0.0794, -0.0710, -0.1225, 0.1099, 0.2953, -0.0559, -0.1485, 0.0680, -0.2965, 0.2385, 0.0343, 0.2537, -0.0177, -0.1222, -0.0282, 0.1011, 0.1323, -0.2209, -0.0770, 0.1612, -0.1423, -0.1570, 0.1201, 0.1101, 0.1283, 0.1167, 0.0975, -0.0207, 0.0825, 0.0321, -0.0268, 0.0908, -0.1218, 0.0625, -0.0566, 0.0422, -0.3426, -0.1474, -0.1277, 0.0342, 0.0112]
  ];
var FA_SCALE = {mu_min: -0.03402, mu_max: 0.04129, sd_min: 0.00736, sd_max: 0.16771};
// similarity: background p50 0.105 p90 0.333 p99 0.596; target p10 0.384 p50 0.599

var KNOWN_PLOTS = [
    [106.34885, 38.58912],
    [118.08482, 39.79238],
    [118.00819, 33.42989],
    [115.19707, 35.35014],
    [119.01767, 37.07411],
    [106.52935, 37.92940],
    [116.52706, 36.37760],
    [114.31705, 35.94833],
    [118.19695, 33.89111],
    [117.45127, 38.90228],
    [116.58722, 39.94995],
    [114.47467, 37.24097],
    [118.28807, 39.70074],
    [101.66812, 36.65315],
    [106.67264, 10.00604],
    [36.22930, 34.10650],
    [10.83056, 59.73626],
    [10.86873, 43.91640],
    [16.68483, 40.24328],
    [4.35002, 43.78172],
    [71.80046, 29.54215],
    [-85.47234, 30.20893],
    [77.40417, 13.26138],
    [78.13298, 13.05371],
    [14.98451, 36.73982],
    [0.21842, 35.73780],
    [2.77899, 36.65999],
    [5.20957, 34.72016],
    [2.71284, 36.58902],
    [0.31387, 36.14756],
    [2.69169, 36.45635],
    [32.56049, 15.45717],
    [-58.80560, -34.54301],
    [-60.32799, -31.36369],
    [41.59734, 37.85047],
    [130.25049, 47.23264],
    [123.23829, 44.71365],
    [125.43291, 46.32823],
    [121.33465, 31.00214],
    [120.97869, 30.90402],
    [121.62173, 30.00084],
    [24.78688, -27.72297],
    [21.64910, 61.63684],
    [6.48990, 60.65022],
    [-35.74784, -9.63510],
    [115.95895, -32.46670]
  ];

// ===========================================================================
// 1. SOURCES
// ===========================================================================
var AEF = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');
var DW  = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1');

function embedding(year) {
  return AEF.filterDate(year + '-01-01', (year + 1) + '-01-01')
            .filterBounds(AOI).mosaic().clip(AOI);
}

// SELF-CHECK. Run once and read the console: bands must be A00..A63, and the
// norm of a 2018 embedding must be ~1.0. The unit-norm property is what makes
// a dot product equal a cosine below -- if it ever stops holding, every
// similarity number in this script is wrong and nothing will say so.
var e18 = embedding(YEAR_A);
var e24 = embedding(YEAR_B);
print('AlphaEarth bands (expect A00..A63):', e18.bandNames());
print('|embedding| at AOI centroid (expect ~1.0):',
      e18.pow(2).reduce(ee.Reducer.sum()).sqrt()
         .reduceRegion({reducer: ee.Reducer.first(),
                        geometry: AOI.centroid(1000), scale: 10}));

// ===========================================================================
// 2. THE CHANGE VECTOR, AND SIMILARITY TO THE 46 PLOTS
// ===========================================================================
// diff = 2024 - 2018, matching extract_embeddings_gee.build_embedding_stack.
// It is NOT unit-norm (the endpoints are), so it must be normalised before a
// dot product means a cosine.
var diff = e24.subtract(e18);
var diffNorm = diff.pow(2).reduce(ee.Reducer.sum()).sqrt().max(1e-9);
var dUnit = diff.divide(diffNorm);

// AlphaEarth's band names, rebuilt in plain JS. ee.Image.constant() names its
// bands 'constant', 'constant_1', ... and Earth Engine matches multi-band
// operands BY NAME -- an unnamed constant image would not align with A00..A63,
// and the failure is silent, not an error.
var BANDS = [];
for (var bi = 0; bi < 64; bi++) { BANDS.push('A' + (bi < 10 ? '0' : '') + bi); }

function cosineTo(vector) {
  return dUnit.multiply(ee.Image.constant(vector).rename(BANDS))
              .reduce(ee.Reducer.sum());
}

// Max over the 3 sub-prototypes: "does this look like ANY of the three modes of
// de-urbanisation in the label set", not "does it look like their average".
// k=3 and k=8 tie within seed noise (top-100 18.4+-2.6 vs 19.2+-1.2); 3 is kept
// because each mode is then backed by ~15 plots and can be inspected.
var similarity = ee.ImageCollection(PROTOTYPES.map(cosineTo)).max()
                   .rename('similarity');

// ===========================================================================
// 3. DYNAMIC WORLD: THE DESTINATION THE EMBEDDING CANNOT SEE
// ===========================================================================
function dwYear(year) {
  return DW.filterDate(year + '-01-01', (year + 1) + '-01-01')
           .filterBounds(AOI).select(['built', 'crops', 'trees', 'grass',
                                      'shrub_and_scrub', 'bare', 'water'])
           .mean().clip(AOI);
}
var dwA = dwYear(YEAR_A);
var dwB = dwYear(YEAR_B);

// De-urbanisation: built-up probability fell. Independent of AlphaEarth.
var deurban = dwA.select('built').subtract(dwB.select('built'))
                 .clamp(0, 1).rename('deurbanisation');

// Destination is cropland, not nature. Two conditions, deliberately ANDed:
// crops rose AND crops is what is actually there in 2024. Either alone admits
// a field that was already cropland, or a noisy single-year rise.
var cropGain = dwB.select('crops').subtract(dwA.select('crops')).clamp(0, 1);
var toCropland = cropGain.multiply(dwB.select('crops')).sqrt()
                         .rename('to_cropland');

// The discriminator the embedding lacks, kept as its own layer so the
// Cropland-vs-Nature call can be inspected rather than trusted.
var natural = dwB.select('trees').add(dwB.select('grass'))
                 .add(dwB.select('shrub_and_scrub'));
var cropVsNature = dwB.select('crops')
                      .subtract(natural).rename('crop_vs_nature');

// ===========================================================================
// 4. THE OTHER ACQUISITION METRICS (src/acquisition.py, in Earth Engine)
// ===========================================================================
// novelty_to_reference: 1 - cosine to the nearest labelled plot. Exact novelty
// needs all 6,490 plots; 24 k-means centroids of the label set stand in, which
// is an approximation and is why it carries weight 0 by default.
var novelty = ee.Image(1).subtract(
    ee.ImageCollection(NOVELTY_CENTROIDS.map(cosineTo)).max()).rename('novelty');

// feature_activation (core-set paper eq. 2). mu and sigma MUST be scaled to
// (0,1] before gamma is formed -- once sigma > 1 the log turns positive and the
// ranking inverts against the method's own premise. Constants are the labelled
// frame's range; AlphaEarth is signed, not ReLU, so treat FA here as untested.
var mu = diff.reduce(ee.Reducer.mean());
var sd = diff.reduce(ee.Reducer.stdDev());
function unitScale(img, lo, hi) {
  return img.subtract(lo).divide(hi - lo).clamp(1e-6, 1.0);
}
var muS = unitScale(mu, FA_SCALE.mu_min, FA_SCALE.mu_max);
var sdS = unitScale(sd, FA_SCALE.sd_min, FA_SCALE.sd_max);
// Rename explicitly: ee.Image(1).subtract(x) keeps the CONSTANT's band name,
// so without this the percentile keys below are constant_p1, not gamma_p1, and
// the ee.Number(...) lookups silently return null.
var gamma = ee.Image(1).subtract(muS).multiply(sdS.log()).multiply(-1)
              .rename('gamma');
// min-max over the pool, as in acquisition.py -- here the pool is the AOI.
var gRange = gamma.reduceRegion({
  reducer: ee.Reducer.percentile([1, 99]), geometry: AOI,
  scale: 1000, maxPixels: 1e10, bestEffort: true});
var activation = ee.Image(1).subtract(
    gamma.subtract(ee.Number(gRange.get('gamma_p1')))
         .divide(ee.Number(gRange.get('gamma_p99'))
                 .subtract(ee.Number(gRange.get('gamma_p1'))))
         .clamp(0, 1)).rename('activation');

// label_complexity: entropy of the Dynamic World class mix in a cell. The
// nodata/ignored classes are excluded, as the paper does -- counted as a class,
// nodata reads as variety and an empty cell scores as one of the most complex.
var lcBands = ['built', 'crops', 'trees', 'grass', 'shrub_and_scrub'];
var lcTotal = dwB.select(lcBands).reduce(ee.Reducer.sum()).max(1e-9);
var lcParts = lcBands.map(function (b) {
  var p = dwB.select(b).divide(lcTotal).max(1e-9);
  // Common band name: ImageCollection.sum() reduces across images per band
  // NAME, so five differently-named bands would come back as five bands
  // rather than as their sum.
  return p.multiply(p.log()).rename('h');
});
var labelComplexity = ee.ImageCollection(lcParts).sum()
    .multiply(-1).divide(Math.log(lcBands.length)).rename('label_complexity');

// NOT computable here, stated so nobody looks for them:
//  * model entropy / margin / BALD / conformal set size -- these read the
//    deployed torch model's posterior, which does not exist in Earth Engine.
//    They belong in stage 2, after infer_patches.py maps the shortlist.
//  * vendi_score -- needs an eigendecomposition. Run it in Python on the
//    exported candidates (acquisition.vendi_score) to check the batch is not
//    1,250 pictures of the same field.

// ===========================================================================
// 5. THE COMPOSITE ACQUISITION SCORE
// ===========================================================================
var acquisition = similarity.multiply(W.similarity)
    .add(toCropland.multiply(W.destination))
    .add(novelty.multiply(W.novelty))
    .divide(W.similarity + W.destination + W.novelty)
    .rename('acquisition');

// Eligibility BEFORE ranking. A cell that does not clear the similarity floor
// cannot supply this class however novel it is -- ranking without the filter
// selects spectacular terrain that contains nothing wanted.
var eligible = similarity.gte(SIM_MIN).and(deurban.gt(0.05));
var acquisitionMasked = acquisition.updateMask(eligible);

// ===========================================================================
// 6. AGGREGATE TO THE 5 KM LABELLING UNIT
// ===========================================================================
// PROJECTIONS -- read this before changing anything here.
//
// `mosaic()` and `mean()` DROP the default projection. Every image in this
// script descends from one of them, so they are all projection-less, and
// reduceResolution refuses a projection-less input:
//     "The input to reduceResolution does not have a valid default projection."
// The fix is to reassert the native grid before aggregating. It has to be done
// at the point of use, not at the source, because the arithmetic in sections
// 2-5 (and ee.Image.constant in particular) drops it again.
var NATIVE = ee.Projection('EPSG:4326').atScale(10);

// Note the approximation this accepts: a 10 m geographic pixel is not 10 m in
// x away from the equator, so the number of input pixels per output cell grows
// with latitude. maxPixels below carries enough headroom for that (the true
// caps are 2,500 and 100 at the equator). It is fine for a *ranking prior* and
// would not be fine for anything area-denominated -- the patch geometry in
// global_patches.py uses per-patch UTM for exactly that reason.
function toCells(img, reducer) {
  reducer = reducer || ee.Reducer.mean();
  var mid = img.setDefaultProjection(NATIVE)
               .reduceResolution({reducer: reducer, maxPixels: 10000})
               .reproject({crs: 'EPSG:4326', scale: 500});
  return mid.reduceResolution({reducer: reducer, maxPixels: 500})
            .reproject({crs: 'EPSG:4326', scale: CELL_M});
}

// How much CANDIDATE GROUND the cell holds, in hectares. This is the band that
// decides whether a cell is labellable at all, and it is the same rule
// PATCH_SAMPLING.md section B uses: a class occupying 40 scattered pixels
// cannot be labelled however high it scores. mean() of a 0/1 mask is the
// eligible fraction; x 2500 turns it into hectares of a 25 km2 cell.
var eligibleHa = toCells(eligible.unmask(0)).multiply(2500).rename('eligible_ha');

// Similarity is aggregated at the 90th PERCENTILE, not the mean. A 5 km cell
// holding one bright 200 m conversion is exactly the cell worth an afternoon,
// and its mean similarity is near background. Same reasoning as novelty_p90 in
// the pilot: the pocket of interesting land inside an ordinary patch is the
// thing being bought.
//
// Honest about what this is: p90 is applied at EACH of the two hops, so it is
// the 90th percentile of the 500 m blocks' 90th percentiles, not the true p90
// over the ~250,000 pixels, and it is biased high. It stays because it is
// monotone in "does a strong pocket exist here", which is the only thing it is
// used for -- do not read the value itself as a similarity, and do not compare
// it against the SIM_MIN calibration, which is a per-pixel number.
var simCell = toCells(similarity, ee.Reducer.percentile([90]))
                .rename('similarity_p90');

var cells = toCells(acquisition.updateMask(eligible).unmask(0))
    .rename('acquisition')
    .addBands(simCell)
    .addBands(eligibleHa)
    .addBands(toCells(deurban).rename('deurbanisation'))
    .addBands(toCells(toCropland).rename('to_cropland'))
    .addBands(toCells(cropVsNature).rename('crop_vs_nature'))
    .addBands(toCells(novelty).rename('novelty'))
    .addBands(toCells(labelComplexity).rename('label_complexity'))
    .addBands(ee.Image.pixelLonLat());

// ===========================================================================
// 7. DISPLAY
// ===========================================================================
Map.centerObject(AOI, 7);
var SIM_VIS = {min: 0.10, max: 0.70,
               palette: ['000004', '3b0f70', '8c2981', 'de4968', 'fe9f6d', 'fcfdbf']};

Map.addLayer(e24.select(['A01', 'A16', 'A09']), {min: -0.3, max: 0.3},
             'AlphaEarth 2024 (context)', false);
Map.addLayer(similarity, SIM_VIS, 'similarity to Art->Crop (p50 bg = 0.105)');
Map.addLayer(deurban, {min: 0, max: 0.5, palette: ['000000', 'ff9900']},
             'de-urbanisation (DW built fell)', false);
Map.addLayer(toCropland, {min: 0, max: 0.5, palette: ['000000', '00cc44']},
             'destination = cropland (DW)', false);
Map.addLayer(cropVsNature, {min: -0.6, max: 0.6,
             palette: ['1a9850', 'ffffbf', 'd73027']},
             'cropland vs nature (green = nature)', false);
Map.addLayer(novelty, {min: 0.27, max: 0.75}, 'novelty vs label set', false);
Map.addLayer(activation, {min: 0, max: 1}, 'feature activation', false);
Map.addLayer(labelComplexity, {min: 0, max: 1}, 'label complexity (DW)', false);
Map.addLayer(acquisitionMasked, {min: 0.2, max: 0.9,
             palette: ['2b83ba', 'abdda4', 'ffffbf', 'fdae61', 'd7191c']},
             'ACQUISITION SCORE (eligible only)');

// The 46 plots the prototypes were fitted on. Shown last so they sit on top:
// if the bright pixels are not near them at all, the prototypes did not
// transfer and the map should not be sent to anyone.
var KNOWN = ee.FeatureCollection(KNOWN_PLOTS.map(function (c) {
  return ee.Feature(ee.Geometry.Point(c), {cls: 'Artificial -> Cropland'});
}));
Map.addLayer(KNOWN.style({color: '00ffff', pointSize: 6, width: 2}),
             {}, 'the 46 labelled Art->Crop plots');

var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px'}});
legend.add(ui.Label('Artificial -> Cropland acquisition',
                    {fontWeight: 'bold', fontSize: '14px'}));
legend.add(ui.Label('similarity = cosine of the 2018->2024 AlphaEarth change ' +
                    'vector to the nearest of 3 prototypes fitted on 46 plots.'));
legend.add(ui.Label('Background p50 = 0.105, p99 = 0.596. Target plots: ' +
                    'p10 = 0.384, p50 = 0.599.'));
legend.add(ui.Label('Expect ~26x enrichment, NOT high precision: at the ' +
                    '0.71% base rate even AUC 0.915 gives ~6% precision.'));
legend.add(ui.Label('Top candidates are Art->Nature as often as Art->Crop. ' +
                    'Interpret as a two-class destination call.'));
Map.add(legend);

// ===========================================================================
// 8. ACQUIRE: TOP-N CELLS
// ===========================================================================
// Top-N, not a threshold. The calibration says thresholds are a bad interface
// here -- precision moves from 0.007 to 0.061 across the entire usable range,
// so a threshold cannot buy purity, while a rank can still order the search.
// >= 1 ha of candidate ground, matching the pilot's usability rule. Without it
// the ranking happily returns cells whose whole score comes from a handful of
// scattered pixels that no interpreter can act on.
var candidates = cells.updateMask(cells.select('eligible_ha').gte(MIN_HA))
  .sample({region: AOI, scale: CELL_M, geometries: true,
           dropNulls: true, numPixels: 1e6, seed: 0})
  .sort('acquisition', false)
  .limit(TOP_N);

print('candidate cells (capped at 5000 for print):', candidates.limit(5000).size());
print('first 10 candidates:', candidates.limit(10));

Map.addLayer(candidates.style({color: 'ffffff', pointSize: 4}), {},
             'top-' + TOP_N + ' candidate cells', false);

Export.table.toDrive({
  collection: candidates,
  description: 'art_to_cropland_candidates',
  folder: EXPORT_FOLDER,
  fileFormat: 'CSV',
  selectors: ['longitude', 'latitude', 'acquisition', 'similarity_p90',
              'eligible_ha', 'deurbanisation', 'to_cropland',
              'crop_vs_nature', 'novelty', 'label_complexity']
});

// ===========================================================================
// AFTER THE EXPORT
// ===========================================================================
// 1. Check the batch is diverse before anyone is asked for time:
//        from acquisition import vendi_score
//        vendi_score(embeddings_of_candidates)   # effective distinct places
//    A number far below TOP_N means the surface collapsed onto one landscape.
// 2. Feed the CSV to plan_patch_sampling.py as the candidate frame; it prices
//    the yield per class and thins to one point per 500 m.
// 3. The confirm rate of THIS channel is unmeasured. Until the first batch
//    comes back, every count derived from it is a lower bound -- the same
//    unpriced-channel caveat as PATCH_SAMPLING.md section C. Measure it on the
//    first 100 before committing anyone to 1,250.
