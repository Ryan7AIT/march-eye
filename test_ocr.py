#!/usr/bin/env python3
"""
ocr_compare.py — Extrait le texte d'une image avec plusieurs moteurs OCR
open-source, pour comparer rapidement leurs résultats.

Backends disponibles (2026) :
  - tesseract     : baseline classique, rapide, CPU only, léger
  - paddleocr     : PaddleOCR-VL, meilleur ratio précision/coût actuellement
                    (~95%+ sur OmniDocBench avec seulement 0.9B paramètres)
  - qwen          : Qwen2.5-VL-7B-Instruct, très bon sur documents complexes
                    (tableaux, diagrammes, mise en page non triviale)
  - got           : GOT-OCR2.0, unifié, bon sur formules/tableaux

Installation (installe seulement ce dont tu as besoin) :
    pip install pytesseract pillow                     # tesseract
    sudo apt install tesseract-ocr                      # + binaire système
    pip install paddlepaddle paddleocr                   # paddleocr
    pip install torch transformers accelerate qwen-vl-utils  # qwen
    pip install torch transformers                       # got (via transformers)

Usage:
    python ocr_compare.py photo.jpg --backend tesseract
    python ocr_compare.py photo.jpg --backend paddleocr
    python ocr_compare.py photo.jpg --backend qwen
    python ocr_compare.py photo.jpg --backend all      # lance tous les backends dispos
"""

import argparse
import sys
from pathlib import Path


def ocr_tesseract(image_path: str) -> str:
    import pytesseract
    from PIL import Image

    img = Image.open(image_path)
    return pytesseract.image_to_string(img)


def ocr_paddleocr(image_path: str) -> str:
    import os

    # Paddle 3.x + oneDNN on Windows CPU can crash during predict (PIR/MKLDNN bug).
    os.environ.setdefault("FLAGS_enable_pir_api", "0")

    import paddleocr
    from paddleocr import PaddleOCR

    ver = getattr(paddleocr, "__version__", "2.0.0")
    parts = ver.split(".")
    use_v3 = int(parts[0]) >= 3 if parts and parts[0].isdigit() else False

    if use_v3:
        ocr = PaddleOCR(use_textline_orientation=True, lang="fr", enable_mkldnn=False)
        pages = ocr.predict(image_path)
        lines = []
        for page in pages:
            lines.extend(page["rec_texts"])
        return "\n".join(lines)

    ocr = PaddleOCR(use_angle_cls=True, lang="fr")
    result = ocr.ocr(image_path, cls=True)
    lines = []
    for page in result or []:
        if not page:
            continue
        for line in page:
            text = line[1][0]
            lines.append(text)
    return "\n".join(lines)


def ocr_qwen(image_path: str, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> str:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": "Extract all text visible in this image, exactly as written. "
                    "Output only the raw text, no commentary.",
                },
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(**inputs, max_new_tokens=2048)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output[0].strip()


def ocr_got(image_path: str, model_id: str = "stepfun-ai/GOT-OCR2_0") -> str:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        use_safetensors=True,
    )
    model = model.eval()
    result = model.chat(tokenizer, image_path, ocr_type="ocr")
    return result


BACKENDS = {
    "tesseract": ocr_tesseract,
    "paddleocr": ocr_paddleocr,
    "qwen": ocr_qwen,
    "got": ocr_got,
}


def run_backend(name: str, image_path: str) -> None:
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    try:
        text = BACKENDS[name](image_path)
        print(text if text.strip() else "(aucun texte détecté)")
    except ImportError as e:
        print(f"[skip] dépendances manquantes pour '{name}': {e}")
    except Exception as e:
        print(f"[erreur] '{name}' a échoué: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare plusieurs OCR open-source sur une image.")
    parser.add_argument("image", help="Chemin vers l'image à analyser")
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()) + ["all"],
        default="tesseract",
        help="Backend OCR à utiliser (défaut: tesseract)",
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Fichier introuvable: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.backend == "all":
        for name in BACKENDS:
            run_backend(name, args.image)
    else:
        run_backend(args.backend, args.image)


if __name__ == "__main__":
    main()