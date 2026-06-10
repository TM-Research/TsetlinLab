# TsetlinLab Modular Viewer

This is a structured alternative to the single-file `viewer.html`.

## Files

- `index.html` - layout and tabs
- `styles.css` - styling only
- `config.js` - easy tuning values
- `app.js` - behavior and rendering logic

## Tabs (clear order)

1. **Learned Representation** - one class at a time, concept tree
2. **Model Behavior** - overall behavior and most-used features
3. **Why This Prediction** - per-row explanation

## GitHub Pages

Works on GitHub Pages as static files.

Example URL:

`viewer_modular/index.html?src=../data/cf_nslkdd_atlas.json&csv=../data/cf_nslkdd_predictions.csv`
