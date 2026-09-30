from .my_custom_node import MyCustomNode

# Mapowanie: "NazwaKlasyWSystemie": KlasaPython
NODE_CLASS_MAPPINGS = {
    "MyCustomNode": MyCustomNode
}

# Mapowanie przyjaznych nazw wyświetlanych użytkownikowi w UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCustomNode": "🌟 Mój Własny Custom Node"
}

# Eksport mapowań
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
