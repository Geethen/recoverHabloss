# Round one — the note that goes to the interpreters

Send this once the app is deployed and the b001 rows are cleared (see
`ACTIVE_LEARNING.md` and the checklist at the foot of this file). The two links
differ by one character and they are not interchangeable — `e1` is Dina and `e2`
is Øyvind, and the id is what every saved row is keyed on.

Keep it short when you send it. Everything the app teaches on first run is in
the app; this exists to get someone from an inbox to a first call, and to put
the one rule that is easy to get wrong in front of them before they start.

---

**Subject:** RECOVER land-cover interpretation — your link, and 20 minutes of calibration first

Hi Dina, Hi Øyvind,

We are ready to start the interpretation round. Everything runs in the browser —
nothing to install, no sign-in, and no Earth Engine account needed.

**Your personal link** (they are different — please use your own, and bookmark
it):

* Dina — https://geethen.github.io/recoverHabloss/?expert=e1
* Øyvind — https://geethen.github.io/recoverHabloss/?expert=e2

The link carries who you are, which is how your work is saved to your own
workspace and how we can measure where the two of you agree. If you open the
other person's link your calls are filed under their name, so it is worth
checking the address before the first session.

**Please do the two calibration batches first, in this order:**

1. **Calibration — teaching** (25 points). These have an agreed answer and you
   are shown it after every call. Work them exactly as you would real ones.
2. **Calibration — qualification** (25 points). Same idea, but you are not shown
   the answers until the end.

They are the first two entries in the batch dropdown. Together they take about
20–30 minutes, and they matter more than they look: the point is not to score
anyone, it is to get the two of you calling the same ground the same way before
there is any real data to lose. The report at the end shows *which* pairs of
classes we disagree on, which is the useful part — a shared habit of calling long
fallow "Cropland" is a ten-minute conversation, and it looks nothing like the
same headline number made of scattered one-offs.

**Then the three real batches** — `b001` (100 points), `cov001` (75) and
`rar001` (75). Take them in any order. Your own share of each is shown next to
its name; the app hands you only the points assigned to you, apart from a small
deliberate overlap where we want both readings.

**The one rule worth reading before you start.** You are judging the **yellow
square**, not the point in the middle of it. It is a real 10 m Sentinel-2 pixel,
and the call is **whichever class covers most of it**:

* a house in the corner of an otherwise grassy square → **Nature**
* a square that is two-thirds rooftop → **Artificial**

The square is deliberately not centred on the marker — it is a pixel of the
actual satellite grid, which is the same grid the model is trained and run on.
The red ring is only there to help you find it and fades out as you zoom in.

Three smaller things:

* **Confidence (1 / 2 / 3) is required** on every real call. It is one keystroke
  and it changes what a disagreement between the two of you means later.
* **"Cannot interpret" is a real answer**, not a failure. Cloud, shadow, no
  usable imagery, genuinely ambiguous — say so and move on. It asks for a reason
  and no confidence.
* **Everything is saved as you go**, including offline. If the connection drops,
  keep working; the app delivers the queue when it comes back.

The panel below the map carries the evidence — a nine-year image filmstrip, the
spectral profile, and what the other global land-cover products say about the
point. All of it is pre-computed, so it loads instantly and works with nothing
signed in. The other products are collapsed at the foot on purpose: they are
somebody else's classification and they are worth checking *after* you have made
your own call, not before.

There is a short brief on first run and a legend behind the **ⓘ** on the map.
Any question at all — especially "is this the kind of thing you meant?" during
calibration — please just ask. That is exactly what the calibration round is for.

Thanks both,
Geethen

---

## Before this goes out

- [ ] **Clear the 14 `b001` rows in the Sheet.** They are the previous pair's
      trial calls under `e1` (4), `e2` (9) and `Geethen` (1). `e1` and `e2` now
      mean Dina and Øyvind, so any row left behind is one id holding two people
      — which is the exact failure the annotation key exists to prevent, and it
      is silent. Verify with
      `curl '<exec-url>?action=status&token=<token>'`: `b001` should be gone or
      report `n: 0`.
- [ ] **Update the `LABEL_APP_CONFIG_JS` Actions secret** to the new roster.
      `app/config.js` governs a local serve only; the published app is built
      from the secret, and the two are kept in step by hand. If you skip this
      the app deploys with the old names.
- [ ] **Merge to `main`.** Pages builds from `main`, so the links above 404
      until it does.
- [ ] Open both links yourself and confirm each shows its own name in the header
      and a non-zero count next to the batches.
