import torch

class MyCustomNode:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        """
        Definiuje porty wejściowe dla węzła w interfejsie ComfyUI.
        """
        return {
            "required": {
                "image": ("IMAGE",),  # Wejście obrazu (Tensor)
                "text_input": ("STRING", {
                    "multiline": False, 
                    "default": "Wpisz coś tutaj"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.1
                }),
            },
        }

    # Definiuje typy zwracane przez węzeł (zawsze jako krotka/tuple)
    RETURN_TYPES = ("IMAGE", "STRING")
    # Nazwy wyświetlane na portach wyjściowych
    RETURN_NAMES = ("IMAGE", "TEXT")

    # Nazwa funkcji w klasie, która zostanie wykonana
    FUNCTION = "execute_logic"

    # Kategoria, w której węzeł pojawi się w menu pod prawym przyciskiem myszy
    CATEGORY = "MojeWęzły"

    def execute_logic(self, image, text_input, strength):
        """
        Główna logika przetwarzania danych.
        """
        # Przykład prostej operacji na obrazie (mnożenie przez strength)
        # W ComfyUI obrazy to Tensory o kształcie [B, H, W, C] w zakresie 0.0 - 1.0
        modified_image = image * strength
        modified_image = torch.clamp(modified_image, 0.0, 1.0)

        processed_text = f"Przetworzono tekst: {text_input} z siłą {strength}"

        return (modified_image, processed_text)
