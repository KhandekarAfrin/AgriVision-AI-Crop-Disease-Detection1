# Crop Disease Detection

Streamlit app that classifies one uploaded leaf image into one of 38 PlantVillage classes using a pretrained MobileNetV2 model. It shows whether the image matches a healthy or disease class, the prediction confidence, prevention guidance, and alternative classes.

Screenings are saved locally in `prediction_history.db`, and uploaded images are saved in `history_images/`. The app includes an in-page history view and a ZIP export with a CSV and the saved images. No uploaded image is sent to the history service; files remain in this project folder.

Model: [`neghani/plantvillage-mobilenetv2`](https://huggingface.co/neghani/plantvillage-mobilenetv2). The model is downloaded from Hugging Face on first launch and cached for later runs, so internet is needed the first time.

## Windows setup

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks virtual environment activation, skip the activation line and run install/start using `.venv\Scripts\python.exe -m pip ...` and `.venv\Scripts\python.exe -m streamlit run app.py`.

## Dataset scope

This model knows 38 PlantVillage classes across 14 crops. It is appropriate for images from that dataset. Its confidence is a softmax score, not the probability that the diagnosis is medically/agronomically correct. Unknown diseases can be forced into a known class. Confirm treatment decisions with a local agriculture expert.
