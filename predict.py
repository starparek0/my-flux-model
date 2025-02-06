from PIL import Image

class Predictor:
    def predict(
        self,
        prompt: str,
        extra_lora: str,
        image: Image.Image,  # Cog automatycznie konwertuje wejście na obiekt PIL.Image
        mask: Image.Image,   # Podobnie dla maski
        output_format: str = "png"
    ) -> Image.Image:
        """
        Generuje obraz na podstawie podanego prompta, wagi LoRA, obrazu wejściowego oraz maski.
        Dla demonstracji funkcja kopiuje obraz wejściowy i nakłada maskę.
        
        W rzeczywistej implementacji należy zastąpić część kopiowania wywołaniem modelu generującego.
        """
        # Konwersja obrazu wejściowego
        input_image = image.convert("RGB")
        input_mask = mask.convert("L")
        
        # !!! TU WPISZ WYWOŁANIE SWOJEGO MODELU !!!
        # Dla celów demonstracyjnych kopiujemy oryginalny obraz:
        generated_image = input_image.copy()
        
        # Nałożenie maski – zastąpienie fragmentu oryginalnego wygenerowanym obrazem tam, gdzie maska ma wartość 255
        final_image = self.apply_mask(input_image, generated_image, input_mask)
        
        return final_image

    def apply_mask(
        self,
        original: Image.Image,
        generated: Image.Image,
        mask: Image.Image
    ) -> Image.Image:
        """
        Łączy oryginalny obraz z wygenerowanym fragmentem na podstawie maski.
        Tam, gdzie maska ma wartość 255, zastępuje oryginał wygenerowanym fragmentem.
        """
        # Dopasowanie rozmiarów obrazów
        if original.size != generated.size:
            generated = generated.resize(original.size)
        if original.size != mask.size:
            mask = mask.resize(original.size)
        
        # Konwersja maski do trybu "L" (skala szarości) i binaryzacja przy progu 128
        mask = mask.convert("L")
        threshold = 128
        mask = mask.point(lambda p: 255 if p > threshold else 0)
        
        # Łączenie obrazów – tam, gdzie maska wynosi 255, wybieramy wygenerowany obraz
        final_image = Image.composite(generated, original, mask)
        return final_image
