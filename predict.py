from diffusers import StableDiffusionInpaintPipeline
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
import cog
from pathlib import Path

class Predictor(cog.Predictor):
    def setup(self):
        """Wykonuje się tylko raz podczas uruchamiania aplikacji."""
        self.pipeline = None  # Pipeline będzie ładowany dynamicznie

    @cog.input("prompt", type=str, help="Tekstowy opis obrazu do wygenerowania.")
    @cog.input("image", type=Path, help="Obraz wejściowy (bazowy).")
    @cog.input("mask", type=Path, help="Maska do retuszu (obszar do zmiany).")
    @cog.input("lora_url", type=str, default="", help="Link do modelu LoRA na Hugging Face (opcjonalne).")
    def predict(self, prompt: str, image: Path, mask: Path, lora_url: str) -> Path:
        """Generuje nowy obraz na podstawie wejść."""

        # Ładujemy bazowy model, jeśli jeszcze nie został załadowany
        if not self.pipeline:
            print("Ładowanie bazowego modelu...")
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "flux-dev/inpainting",
                torch_dtype=torch.float16,
                safety_checker=None
            ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Jeśli podano lora_url, pobieramy i ładowamy model LoRA
        if lora_url:
            try:
                print(f"Ładowanie LoRA z {lora_url}...")
                lora_path = hf_hub_download(repo_id=lora_url.split("/")[4], filename=lora_url.split("/")[-1])
                self.pipeline.load_lora_weights(lora_path)
            except Exception as e:
                print(f"Błąd podczas ładowania LoRA: {e}")

        # Ładujemy obrazy
        input_image = Image.open(image).convert("RGB")
        mask_image = Image.open(mask).convert("L")  # Maska musi być w skali szarości

        # Generujemy wynik
        result = self.pipeline(prompt=prompt, image=input_image, mask_image=mask_image).images[0]

        # Zapisujemy wynik do pliku tymczasowego
        output_path = Path("output.png")
        result.save(output_path)

        return output_path