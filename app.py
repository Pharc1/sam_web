from flask import Flask, request, jsonify
import numpy as np
import onnxruntime
from PIL import Image
import io
import logging

app = Flask(__name__)

# Configuration de logging
logging.basicConfig(level=logging.INFO)

# Charger le modèle ONNX
onnx_model_path = "interactive_module_quantized_592547_2023_03_20_sam6_long_all_masks_extra_data_with_ious.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)
logging.info("Modèle ONNX chargé avec succès.")

# Fonction pour prétraiter l'image
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        logging.info("Image ouverte avec succès.")
        image = image.resize((1024, 1024))  # Redimensionne l'image
        logging.info("Image redimensionnée à (1024, 1024).")
        return np.array(image)
    except Exception as e:
        logging.error(f"Erreur lors du prétraitement de l'image : {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Étape 1 : Récupérer les données envoyées
        logging.info("Réception des données de la requête.")
        image_file = request.files['image']
        input_points = request.json['points']
        input_labels = request.json['labels']

        # Étape 2 : Prétraitement de l'image
        image = preprocess_image(image_file.read())

        # Étape 3 : Exemple d'encodage de l'image
        logging.info("Encodage de l'image en cours.")
        image_embedding = np.random.rand(1, 256, 64, 64).astype(np.float32)  # Remplace par get_image_embedding
        logging.info("Encodage de l'image terminé.")

        # Étape 4 : Préparer les données pour le modèle ONNX
        onnx_coord = np.array(input_points).astype(np.float32)[None, :, :]
        onnx_label = np.array(input_labels).astype(np.float32)[None, :]
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        orig_im_size = np.array(image.shape[:2], dtype=np.float32)
        logging.info("Données préparées pour le modèle ONNX.")

        # Étape 5 : Exécuter le modèle
        logging.info("Exécution du modèle ONNX.")
        masks, _, _ = ort_session.run(None, {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": orig_im_size
        })
        logging.info("Exécution du modèle terminée.")

        # Étape 6 : Post-traitement
        masks = (masks > 0).astype(np.uint8).tolist()
        logging.info("Post-traitement terminé.")

        return jsonify({"status": "success", "step": "prediction completed", "masks": masks})

    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
