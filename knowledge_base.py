"""
knowledge_base.py
=================
Builds and queries a ChromaDB vector store populated with rich textual
descriptions of 15 art movements.

Each movement is stored as multiple overlapping chunks so that retrieval
can surface the most relevant passage (period, artists, technique, etc.)
even when the query is narrow.

Usage:
    kb = ArtKnowledgeBase()
    kb.build()                              # idempotent; skips if already built
    results = kb.query("Impressionism", n_results=4)
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(meta["movement"], "—", doc[:120])
"""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

# ── Raw art movement knowledge ────────────────────────────────────────────────
# Each entry has a "movement" key and a list of "chunks" (separate paragraphs
# that will each become an independent ChromaDB document).
ART_KNOWLEDGE: List[Dict[str, Any]] = [
    {
        "movement": "Impressionism",
        "chunks": [
            (
                "Impressionism emerged in France in the 1860s–1870s as a radical "
                "departure from the rigid academic tradition. The name was coined "
                "mockingly by a critic after Claude Monet's painting 'Impression, "
                "Sunrise' (1872). Impressionist artists rejected the precise, "
                "polished finish of Salon painting in favour of quick, visible "
                "brushstrokes that captured a fleeting moment."
            ),
            (
                "Key Impressionist artists include Claude Monet, Pierre-Auguste "
                "Renoir, Edgar Degas, Camille Pissarro, Alfred Sisley, and Berthe "
                "Morisot. Monet is especially known for his series paintings — "
                "Water Lilies, Haystacks, Rouen Cathedral — exploring how the same "
                "subject transforms under changing light and seasons."
            ),
            (
                "Technically, Impressionism is characterised by en plein air "
                "(outdoor) painting, a bright palette, short broken brushstrokes, "
                "and an emphasis on light and atmosphere rather than outline. "
                "Everyday modern life — cafés, parks, ballet, boating — replaced "
                "historical and mythological subjects. Shadows were painted in "
                "colour rather than black or grey."
            ),
            (
                "The Impressionists organised eight independent group exhibitions "
                "between 1874 and 1886, bypassing the official Paris Salon. Their "
                "approach influenced Post-Impressionism, Fauvism, and early "
                "abstraction, making it one of the most pivotal movements in "
                "Western art history."
            ),
        ],
    },
    {
        "movement": "Baroque",
        "chunks": [
            (
                "Baroque art flourished across Europe from roughly 1600 to 1750, "
                "originating in Rome as part of the Counter-Reformation. The "
                "Catholic Church commissioned dramatic, emotionally charged works "
                "to reassert its authority and draw worshippers back from "
                "Protestantism. Baroque art was a direct instrument of persuasion."
            ),
            (
                "Defining characteristics include chiaroscuro (extreme contrast of "
                "light and shadow, pioneered by Caravaggio), dynamic diagonal "
                "compositions, theatrical gesture, and an overwhelming sense of "
                "movement and grandeur. Ceilings became canvases for sweeping "
                "illusionistic frescoes that seemed to open the heavens."
            ),
            (
                "Key Baroque masters: Caravaggio (Italy) — dramatic realism and "
                "tenebrism; Rembrandt van Rijn (Dutch Republic) — psychological "
                "depth in portraiture; Peter Paul Rubens (Flanders) — exuberant "
                "figural compositions; Johannes Vermeer (Dutch Republic) — intimate "
                "interiors bathed in diffuse light; Diego Velázquez (Spain) — "
                "court portraiture of extraordinary naturalism."
            ),
            (
                "Baroque extended beyond painting into sculpture (Bernini's "
                "ecstatic figures) and architecture (St Peter's Square, Versailles). "
                "In music, Bach, Handel, and Vivaldi are its contemporaries. The "
                "Dutch Golden Age, with its merchant-funded genre scenes and still "
                "lifes, represents a secular, Protestant variant of Baroque culture."
            ),
        ],
    },
    {
        "movement": "Renaissance",
        "chunks": [
            (
                "The Renaissance ('rebirth') began in 14th-century Florence and "
                "spread across Europe by the 16th century. Patrons such as the "
                "Medici family funded artists to revive the ideals of ancient "
                "Greece and Rome. It marked the transition from the medieval world "
                "to early modernity, placing humanity — rather than God — at the "
                "centre of intellectual and artistic inquiry."
            ),
            (
                "Key Renaissance innovations: linear perspective (Brunelleschi, "
                "Alberti), anatomical accuracy through direct study of the human "
                "body (Leonardo da Vinci's dissections), sfumato (Leonardo's soft "
                "gradations of tone), and the idealised human form drawn from "
                "classical sculpture. Oil paint, mastered in the North, allowed "
                "unprecedented colour depth."
            ),
            (
                "Central figures: Leonardo da Vinci (The Last Supper, Mona Lisa), "
                "Michelangelo (Sistine Chapel ceiling, David), Raphael (The School "
                "of Athens), Sandro Botticelli (The Birth of Venus), and Jan van "
                "Eyck in the Northern Renaissance. The High Renaissance (c.1490–"
                "1527) produced the most ambitious works before the sack of Rome "
                "disrupted the Papal court."
            ),
            (
                "Humanism — the philosophical movement that emphasised human "
                "potential, reason, and classical learning — underpinned "
                "Renaissance art. Portraits celebrated individual identity; "
                "mythology returned as a legitimate subject; architecture drew on "
                "Roman columns, domes, and symmetry. The printing press "
                "accelerated the spread of Renaissance ideas across Europe."
            ),
        ],
    },
    {
        "movement": "Romanticism",
        "chunks": [
            (
                "Romanticism arose in the late 18th century as a reaction against "
                "Enlightenment rationalism and the Industrial Revolution's "
                "dehumanising effects. Romantic artists exalted emotion, "
                "imagination, and nature's sublime power over reason and order. "
                "It was as much a literary and musical movement as a visual one — "
                "Goethe, Byron, Keats, and Beethoven were its kindred spirits."
            ),
            (
                "Key Romantic painters: Eugène Delacroix (France) — Liberty "
                "Leading the People, vibrant colour and turbulent scenes; Caspar "
                "David Friedrich (Germany) — solitary figures before vast, "
                "misty landscapes conveying existential awe; J.M.W. Turner "
                "(Britain) — storm, fire, and light dissolving solid form; "
                "Francisco Goya (Spain) — moving from court elegance to the "
                "nightmarish Black Paintings."
            ),
            (
                "Romantic painting favoured exotic and historical subjects — "
                "medieval legends, the Orient, revolutions, natural disasters. "
                "Landscapes were no longer backdrops but protagonists: the "
                "Alps, stormy seas, and ancient ruins expressed the 'sublime' — "
                "a mix of awe and terror before nature's overwhelming force. "
                "Nationalism also surged, with artists celebrating folk culture "
                "and national myths."
            ),
        ],
    },
    {
        "movement": "Cubism",
        "chunks": [
            (
                "Cubism, developed by Pablo Picasso and Georges Braque between "
                "1907 and the 1920s, shattered centuries of Western perspective. "
                "Influenced by Cézanne's geometric analysis of form and by "
                "African and Iberian sculpture, Cubism presented multiple "
                "viewpoints simultaneously on a single flat canvas, fragmenting "
                "objects into interlocking planes."
            ),
            (
                "Analytic Cubism (1908–1912) used a near-monochrome palette of "
                "greys and browns to focus attention on formal structure. "
                "Synthetic Cubism (from 1912) introduced collage — newspaper "
                "clippings, wallpaper, sand — making the texture of everyday "
                "material part of the artwork. Juan Gris refined Synthetic Cubism "
                "with more vivid colour and architecturally precise compositions."
            ),
            (
                "Cubism's influence was seismic. It fed into Futurism, "
                "Constructivism, De Stijl, and abstraction broadly. Picasso's "
                "'Les Demoiselles d'Avignon' (1907), though proto-Cubist, is "
                "often cited as the turning point that made 20th-century art "
                "possible. The movement also disrupted sculpture, architecture, "
                "literature (Gertrude Stein), and film editing."
            ),
        ],
    },
    {
        "movement": "Surrealism",
        "chunks": [
            (
                "Surrealism was founded by poet André Breton in Paris in 1924, "
                "drawing heavily on Freudian psychoanalysis and the Dada "
                "movement's anti-rational spirit. Surrealists sought to unlock "
                "the unconscious mind — through dreams, automatic writing, and "
                "chance operations — to reveal a 'superior reality' beneath "
                "everyday appearances."
            ),
            (
                "Two main painting tendencies emerged: illusionistic Surrealism, "
                "where dream imagery is rendered with hyper-realistic precision "
                "(Salvador Dalí's melting watches in 'The Persistence of Memory', "
                "René Magritte's witty visual paradoxes); and automatist "
                "Surrealism, where gestural or chance-based mark-making bypasses "
                "conscious control (Joan Miró's biomorphic fields, Max Ernst's "
                "frottage and decalcomania)."
            ),
            (
                "Key Surrealists: Salvador Dalí (Spain/USA), René Magritte "
                "(Belgium), Frida Kahlo (Mexico — though she resisted the label), "
                "Giorgio de Chirico (Italy), Meret Oppenheim (Switzerland), and "
                "Man Ray (USA/France). Surrealism influenced cinema (Buñuel), "
                "fashion (Elsa Schiaparelli), and later Abstract Expressionism, "
                "pop culture, and advertising imagery."
            ),
        ],
    },
    {
        "movement": "Abstract Expressionism",
        "chunks": [
            (
                "Abstract Expressionism emerged in New York in the 1940s–1950s, "
                "making it the first major American art movement to gain "
                "international influence. Many of its practitioners were European "
                "émigrés who fled World War II, bringing Surrealist automatism "
                "and existentialist philosophy into a new American context."
            ),
            (
                "Two main strands: Action Painting (the gesture and physical act "
                "of painting as the subject — Jackson Pollock's drip paintings, "
                "Willem de Kooning's slashing brushwork, Franz Kline's black-and-"
                "white calligraphic gestures) and Color Field Painting (large "
                "areas of flat, luminous colour evoking emotional or spiritual "
                "states — Mark Rothko's hovering rectangles, Barnett Newman's "
                "vertical 'zips', Helen Frankenthaler's stained canvases)."
            ),
            (
                "Critically, Abstract Expressionism was championed by critic "
                "Clement Greenberg, who saw it as the purest expression of "
                "painting's essential flatness. It influenced Minimalism, "
                "Conceptual Art, and Neo-Expressionism. The movement also "
                "marked a shift in the global art world's centre from Paris "
                "to New York."
            ),
        ],
    },
    {
        "movement": "Pop Art",
        "chunks": [
            (
                "Pop Art emerged independently in Britain (Richard Hamilton, "
                "Eduardo Paolozzi) in the mid-1950s and exploded in the USA "
                "(Andy Warhol, Roy Lichtenstein, Jasper Johns, Robert "
                "Rauschenberg) in the early 1960s. It directly challenged "
                "fine-art hierarchies by embracing mass-produced commercial "
                "imagery: advertising, comic strips, celebrity photographs, "
                "and supermarket packaging."
            ),
            (
                "Andy Warhol elevated consumer icons — Campbell's Soup Cans, "
                "Brillo Boxes, Marilyn Monroe screenprints — into art objects, "
                "questioning originality and the artist's role. Roy Lichtenstein "
                "reproduced comic-book panels with Ben-Day dots at monumental "
                "scale. Jasper Johns painted American flags and targets, "
                "blurring the line between image and object."
            ),
            (
                "Pop Art's visual vocabulary — bold outlines, flat primaries, "
                "repetition, irony — permeated graphic design, fashion, music "
                "packaging (Warhol's Velvet Underground banana cover), and "
                "advertising. It anticipated postmodern debates about "
                "simulation, spectacle, and the commodification of culture, "
                "and directly preceded Conceptual Art and Neo-Pop."
            ),
        ],
    },
    {
        "movement": "Expressionism",
        "chunks": [
            (
                "Expressionism developed in Germany and Austria in the early "
                "20th century (c.1905–1925), seeking to depict subjective "
                "emotional experience rather than objective physical reality. "
                "Form and colour were distorted or exaggerated to express "
                "anxiety, alienation, desire, or spiritual yearning. It was "
                "a direct reaction against Impressionism's detached observation."
            ),
            (
                "Key groups and artists: Die Brücke (The Bridge), founded in "
                "Dresden 1905 — Ernst Ludwig Kirchner's jagged Berlin street "
                "scenes; Der Blaue Reiter (The Blue Rider), Munich — Wassily "
                "Kandinsky moving towards pure abstraction, Franz Marc's "
                "vivid animal paintings. Edvard Munch's 'The Scream' (1893) "
                "is a proto-Expressionist icon. Egon Schiele pushed the figure "
                "into raw, contorted anguish."
            ),
            (
                "Expressionism extended into theatre, literature (Kafka), and "
                "German Expressionist cinema (The Cabinet of Dr Caligari, Nosferatu) "
                "with its distorted sets and harsh lighting. The movement was "
                "condemned and suppressed by the Nazi regime as 'degenerate art' "
                "in 1937, scattering its practitioners. Neo-Expressionism "
                "revived its spirit in the 1980s."
            ),
        ],
    },
    {
        "movement": "Neoclassicism",
        "chunks": [
            (
                "Neoclassicism flourished from roughly 1750 to 1850, inspired "
                "by the archaeological rediscovery of Pompeii and Herculaneum "
                "and a wider Enlightenment turn toward reason, order, and civic "
                "virtue. It was a conscious reaction against the frivolity of "
                "Rococo, championing the clarity and moral seriousness of ancient "
                "Greece and Rome."
            ),
            (
                "Jacques-Louis David is its central painter: 'Oath of the Horatii' "
                "(1784) exemplifies the style's austere composition, sharp outline, "
                "muted palette, and didactic subject — Roman heroes sacrificing "
                "personal feeling for the republic. David later became the "
                "official painter of Napoleon, adapting classical grandeur "
                "to imperial propaganda."
            ),
            (
                "Other key Neoclassicists: Jean-Auguste-Dominique Ingres, whose "
                "silky academic nudes blended classical form with sensuous "
                "precision; Antonio Canova in sculpture, whose marble figures "
                "of mythological subjects achieve idealised grace. Neoclassicism "
                "shaped architecture (the US Capitol, the British Museum), "
                "interior design (Empire style), and the academic tradition "
                "that dominated art schools into the 20th century."
            ),
        ],
    },
    {
        "movement": "Art Nouveau",
        "chunks": [
            (
                "Art Nouveau (c.1890–1910) was an international movement that "
                "rejected the historicism dominating 19th-century design in "
                "favour of a new visual language drawn from organic forms: "
                "sinuous plant tendrils, insects, female hair, waves, and "
                "flowing lines. Its ambition was to dissolve the boundary "
                "between fine and applied arts — architecture, furniture, "
                "jewellery, typography, and posters were all part of a "
                "unified aesthetic project."
            ),
            (
                "Key figures: Alphonse Mucha (Czech/Parisian) — decorative "
                "poster art featuring Byzantine-haloed women entwined with "
                "flowers; Gustav Klimt (Vienna) — The Kiss and the Beethoven "
                "Frieze, blending symbolism, gold leaf, and erotic decoration; "
                "Victor Horta and Hector Guimard in architecture (the Paris Métro "
                "entrances); René Lalique in glass and jewellery."
            ),
            (
                "Art Nouveau was known by different names across Europe: Jugendstil "
                "in Germany/Austria, Modernisme in Catalonia (Gaudí's Sagrada "
                "Família), Stile Liberty in Italy. It was largely displaced by "
                "the cleaner geometry of Art Deco after World War I, but its "
                "influence persists in graphic design and decorative arts."
            ),
        ],
    },
    {
        "movement": "Minimalism",
        "chunks": [
            (
                "Minimalism emerged in New York in the 1960s–1970s as a reaction "
                "against Abstract Expressionism's emotional intensity and "
                "gestural mark-making. Minimalist artists stripped work to its "
                "essential geometric form, industrial materials, and literal "
                "presence — rejecting illusion, metaphor, and autobiographical "
                "content. The work is what it is."
            ),
            (
                "Key figures: Donald Judd's stacked steel and Plexiglas 'stacks'; "
                "Dan Flavin's fluorescent tube arrangements; Carl Andre's floor "
                "sculptures of industrial bricks and metal plates; Robert Morris's "
                "grey felt hangings; Frank Stella's shaped canvases with "
                "mathematically derived stripe patterns."
            ),
            (
                "Minimalism was deeply influenced by music (John Cage's silence, "
                "La Monte Young's drones) and phenomenology — the viewer's bodily "
                "experience of moving around the work in real space and time "
                "became integral to the work's meaning. It fed into Land Art, "
                "Conceptual Art, and the 'white cube' gallery aesthetic that "
                "defines contemporary art presentation today."
            ),
        ],
    },
    {
        "movement": "Realism",
        "chunks": [
            (
                "Realism arose in France in the 1840s–1860s in direct opposition "
                "to Romanticism's idealisation and Neoclassicism's classical "
                "subjects. Its manifesto was Gustave Courbet's declaration that "
                "he would paint only what he could see: peasants, stone-breakers, "
                "bathers, the social underbelly of contemporary France. The "
                "mundane and unglamorous became legitimate subjects."
            ),
            (
                "Key Realist painters: Gustave Courbet — 'The Stone Breakers', "
                "'A Burial at Ornans' (monumental canvases given to peasant "
                "subjects previously reserved for history painting); Jean-François "
                "Millet — 'The Gleaners', noble rural labour; Honoré Daumier — "
                "satirical lithographs exposing bourgeois hypocrisy and legal "
                "corruption. American Realism: Winslow Homer, Thomas Eakins."
            ),
            (
                "Realism was shaped by the 1848 revolutions and early socialist "
                "thought — the idea that art should engage with society's real "
                "conditions. Photography's invention in 1839 both challenged and "
                "inspired Realist painting. Realism fed into Impressionism "
                "(which retained the everyday subject but dissolved form in light) "
                "and later Social Realism and Photorealism."
            ),
        ],
    },
    {
        "movement": "Symbolism",
        "chunks": [
            (
                "Symbolism developed in France and Belgium in the 1880s–1890s "
                "as a reaction against both Realism's materialism and "
                "Impressionism's sensory immediacy. Symbolist artists sought to "
                "express ineffable inner states — mystical visions, dreams, "
                "spiritual yearning, erotic anxiety — through symbolic imagery "
                "rather than literal representation. Emotion was evoked, not "
                "depicted."
            ),
            (
                "Key Symbolist painters: Gustave Moreau (France) — jewel-encrusted "
                "mythological scenes with a decadent, hallucinatory quality; "
                "Odilon Redon — floating heads, strange botanical forms, "
                "pastel visions; Fernand Khnopff (Belgium) — enigmatic "
                "sphinxes and androgynous figures; Franz von Stuck (Germany) — "
                "monumental allegories of Sin and Sensuality."
            ),
            (
                "Symbolism was closely allied with Symbolist poetry (Baudelaire, "
                "Verlaine, Mallarmé) and the broader Decadent movement. "
                "It influenced Art Nouveau's decorative symbolism, Expressionism's "
                "psychological intensity, and Surrealism's dream imagery. "
                "The Nabis group (Bonnard, Vuillard) emerged from its orbit."
            ),
        ],
    },
    {
        "movement": "Ukiyo-e",
        "chunks": [
            (
                "Ukiyo-e ('pictures of the floating world') is a genre of "
                "Japanese woodblock print and painting flourishing from the "
                "17th through 19th centuries, centred in the Edo period's "
                "capital. 'Floating world' referred to the transient pleasures "
                "of the urban entertainment districts — kabuki theatre, sumo, "
                "teahouses — depicted with vivid colour and bold outlines."
            ),
            (
                "Master Ukiyo-e artists: Katsushika Hokusai — 'Thirty-Six Views "
                "of Mount Fuji' including the iconic 'Great Wave off Kanagawa', "
                "whose swirling water and stylised foam became one of the most "
                "recognised images in art history; Hiroshige — lyrical landscape "
                "series 'The Fifty-Three Stations of the Tōkaidō'; Kitagawa "
                "Utamaro — intimate okubi-e (large-head) portraits of courtesans."
            ),
            (
                "Ukiyo-e prints arrived in Europe from the 1860s, triggering "
                "Japonisme — a profound influence on the Impressionists "
                "(Monet's Japanese bridge, Degas's cropped compositions, "
                "Mary Cassatt's flat planes) and Post-Impressionists "
                "(Van Gogh copied Hiroshige directly). Ukiyo-e's flat colour "
                "fields, elimination of modelling, and bold cropping "
                "contributed to the 20th century's move away from illusionism."
            ),
        ],
    },
]


# ── Knowledge base class ───────────────────────────────────────────────────────
COLLECTION_NAME = "art_movements"


class ArtKnowledgeBase:
    """
    Manages a ChromaDB collection of art movement knowledge chunks.
    Uses sentence-transformers 'all-MiniLM-L6-v2' for embedding.
    """

    def __init__(self, persist_dir: str = ".chromadb"):
        self._persist_dir = persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, force_rebuild: bool = False) -> None:
        """
        Index all art movement chunks.  Skips if already populated unless
        force_rebuild=True.
        """
        if not force_rebuild and self._collection.count() > 0:
            print(f"[KnowledgeBase] Already built ({self._collection.count()} docs). Skipping.")
            return

        if force_rebuild:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )

        documents, metadatas, ids = [], [], []
        for entry in ART_KNOWLEDGE:
            movement = entry["movement"]
            for i, chunk in enumerate(entry["chunks"]):
                doc_id = f"{movement.lower().replace(' ', '_')}_{i}"
                documents.append(chunk)
                metadatas.append({"movement": movement, "chunk_index": i})
                ids.append(doc_id)

        self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"[KnowledgeBase] Indexed {len(documents)} chunks across {len(ART_KNOWLEDGE)} movements.")

    def query(self, style: str, n_results: int = 4) -> Dict[str, Any]:
        """
        Retrieve the most relevant chunks for the given art style label.

        Args:
            style:     Art movement name (e.g. "Impressionism").
            n_results: How many chunks to retrieve.

        Returns:
            ChromaDB query result dict with keys:
            "documents", "metadatas", "distances", "ids"
        """
        return self._collection.query(
            query_texts=[style],
            n_results=min(n_results, self._collection.count()),
        )