# Dhanuja Presentation Summary

Owner: Dhanuja (Recommender logic + product flow)
Target duration: 3.5 minutes

## 1) What to present

1. What a rule-based recommender means in this project.
2. How user flow works from scan to outfit generation.
3. Why outputs are explainable and not random.
4. How feedback changes future recommendations.

---

## 2) Recommender story in simple language

Our recommender is not guessing randomly.
It follows clear steps:

1. Take valid tops and bottoms from wardrobe.
2. Make possible top-bottom combinations.
3. Score each combination using fixed logic:
   - compatibility,
   - diversity,
   - repeat penalty,
   - user preference signals.
4. Return best options with explanation.

---

## 3) Product flow to explain

Use this flow:

1. Scan one garment image.
2. Item goes to wardrobe.
3. User can review item type/category if needed.
4. Generate outfits for selected days.
5. Mark liked/disliked/worn/skipped.
6. Next generation reflects feedback.

---

## 4) Smart moves and struggles to mention

1. We saw recommendation blocking due to strict eligibility checks.
   - Fixed generation logic so active top+bottom can still generate even if some item types are unknown.

2. We saw repetitive patterns.
   - Added repeat penalties, cooldown logic, and rotation controls.

3. We needed user trust.
   - Added "why this outfit" style explanations and score reasoning.

4. We needed cleaner UX.
   - Better layouts, toasts, navigation, and structured pages for scan/wardrobe/outfits/history/calendar/palette.

---

## 5) Demo script (safe and short)

1. Open scan page, upload one item.
2. Show saved result and useful details.
3. Open wardrobe and show item exists.
4. Open outfits page and generate.
5. Show top-bottom card and explanation.
6. Click feedback action and explain that next recommendations adapt.

---

## 6) 30-second close line

"The recommender is practical because it is clear, controlled, and explainable. We can show exactly why an outfit was chosen and how user feedback changes the next result."
