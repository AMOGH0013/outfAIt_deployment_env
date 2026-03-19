# GitHub Push Checklist (Old Repo)

Use this checklist before final push.

## 1) Verify critical files exist

1. `backend/sumary/MASTER_PRESENTATION_MAP.md`
2. `backend/sumary/amogh/AMOGH_PRESENTATION_SUMARY.md`
3. `backend/sumary/dhanuja/DHANUJA_PRESENTATION_SUMARY.md`
4. `backend/sumary/vaibhavi/VAIBHAVI_PRESENTATION_SUMARY.md`
5. `backend/sumary/reference/*`
6. `backend/PROJECT_FULL_REPORT.md`
7. `backend/EVALUATION & RESULTS - VAIBHAVI/*`

## 2) Recommended commit grouping

1. `docs: add presentation summary pack and ownership map`
2. `docs: sync project and evaluation reports to latest state`

## 3) Safe push flow

```bash
git status
git add backend/sumary backend/PROJECT_FULL_REPORT.md "backend/EVALUATION & RESULTS - VAIBHAVI"
git commit -m "docs: add presentation-ready summary pack and synced reports"
git remote -v
git push origin <your-branch>
```

## 4) Final validation before sharing

1. Open `backend/sumary/README.md`.
2. Confirm each presenter can read only their file and speak without code.
3. Confirm numbers quoted in slides match latest saved artifacts.
