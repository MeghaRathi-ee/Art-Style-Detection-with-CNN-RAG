"""
Query Expansion: converts a CNN class prediction into a rich retrieval query.
"""

# Mapping from class name to expanded search terms
EXPANSION_MAP = {
    "Abstract_Expressionism": "Abstract Expressionism art movement characteristics techniques New York School gestural painting color field action painting Pollock Rothko de Kooning",
    "Action_painting": "Action painting gestural abstraction drip technique spontaneous brushwork Pollock Kline de Kooning canvas arena",
    "Analytical_Cubism": "Analytical Cubism fragmented geometric planes multiple viewpoints monochrome palette Picasso Braque faceted forms",
    "Art_Nouveau_Modern": "Art Nouveau organic flowing lines whiplash curves decorative arts Klimt Mucha Gaudi natural forms ornamental",
    "Baroque": "Baroque painting dramatic chiaroscuro tenebrism emotional intensity Caravaggio Rembrandt Rubens Velazquez grandeur",
    "Color_Field_Painting": "Color Field painting large areas flat color immersive luminous Rothko Newman Frankenthaler soak-stain contemplative",
    "Contemporary_Realism": "Contemporary Realism photorealism hyperrealism accurate realistic depiction modern representational Chuck Close Estes",
    "Cubism": "Cubism geometric fragmentation multiple viewpoints collage Picasso Braque Gris Leger revolutionary representation",
    "Early_Renaissance": "Early Renaissance Quattrocento linear perspective naturalism Masaccio Botticelli Fra Angelico Florence tempera fresco",
    "Expressionism": "Expressionism emotional distortion bold color subjective Die Brucke Der Blaue Reiter Munch Kirchner Kandinsky anxiety",
    "Fauvism": "Fauvism wild beasts vivid non-naturalistic color bold brushwork Matisse Derain Vlaminck pure pigment liberation",
    "High_Renaissance": "High Renaissance Leonardo Michelangelo Raphael sfumato Sistine Chapel Mona Lisa ideal beauty mastery harmony",
    "Impressionism": "Impressionism plein air visible brushstrokes light atmosphere Monet Renoir Degas Pissarro broken color spontaneous",
    "Mannerism_Late_Renaissance": "Mannerism Late Renaissance elongated figures artifice elegance Parmigianino Pontormo El Greco spatial ambiguity virtuosity",
    "Minimalism": "Minimalism geometric simplicity industrial materials objectivity Judd Flavin Andre Stella reduction specific objects",
    "Naive_Art_Primitivism": "Naive art Primitivism self-taught untrained flat perspective bright colors Rousseau Grandma Moses folk art directness",
    "New_Realism": "Nouveau Realisme New Realism found objects assemblage Klein Arman Tinguely everyday materials accumulations",
    "Northern_Renaissance": "Northern Renaissance oil painting meticulous detail Van Eyck Durer Bosch Bruegel symbolism Flemish luminosity",
    "Pointillism": "Pointillism Divisionism small dots pure color optical mixing scientific Seurat Signac Neo-Impressionism luminous",
    "Pop_Art": "Pop Art mass culture advertising consumer products Warhol Lichtenstein Johns Rauschenberg silkscreen Ben-Day dots",
    "Post_Impressionism": "Post-Impressionism Cezanne Van Gogh Gauguin Seurat structure emotion beyond Impressionism geometric symbolic color",
    "Realism": "Realism truthful depiction everyday life Courbet Millet Daumier social observation unidealized contemporary subjects",
    "Rococo": "Rococo ornamental delicate pastel colors elegance Watteau Boucher Fragonard fete galante aristocratic pleasure",
    "Romanticism": "Romanticism emotion sublime nature individual imagination Delacroix Friedrich Turner dramatic landscape passion",
    "Symbolism": "Symbolism dream mystery inner emotion spiritual mythological Moreau Redon Puvis de Chavannes evocative suggestive",
    "Synthetic_Cubism": "Synthetic Cubism collage papier colle bright colors flat shapes mixed media Picasso Braque Gris assembled",
    "Ukiyo_e": "Ukiyo-e Japanese woodblock print floating world Hokusai Hiroshige Utamaro landscape bijin-ga kabuki wave",
}


def expand_query(movement_class: str) -> str:
    """
    Expand a CNN class prediction into a rich retrieval query.
    Falls back to formatting the class name if not in the map.
    """
    return EXPANSION_MAP.get(
        movement_class,
        movement_class.replace("_", " ") + " art movement characteristics techniques artists"
    )
