import re

def clean_granite_text(text: str) -> str:
    """Fix spacing artifacts from Granite Vision OCR output."""
    if not text:
        return ""

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove space before punctuation
    text = re.sub(r'\s([.,;:!?])', r'\1', text)
    # Strip leading/trailing spaces
    text = text.strip()

    return text

from spellchecker import SpellChecker

spell = SpellChecker()
def fix_split_words(text: str) -> str:
    words = text.split()
    corrected = []
    for w in words:
        # if word looks broken, try to fix
        corrected.append(spell.correction(w) or w)
    return ' '.join(corrected)


text = "The plant in the image appears to be a to mat o plant , ident ifiable by the characteristic shape and structure of its leaves . The yellow ing and b row ning with sp ots are indic ative of a common f ung al disease known as se pt oria leaf spot , which is caused by the path ogen Se pt oria ly co pers ici . This disease affects the upper surface of leaves and can lead to significant yield loss in to mat o pl ants . To treat se pt oria leaf spot , the following steps should be taken : 1 . ** Removal of Inf ected Le aves **: Remove all inf ected leaves to prevent the spread of the disease . This is the most effective way to manage the disease . 2 . ** F ung ic ide Application **: Use af ung ic ide that contains the active ingredient cop per o xy ch lor ide or cy an am ide . Follow the manufacturer 's instructions for the appropriate rate and timing of application . 3 . ** Res istance B reed ing **: If possible , use to mat o var iet ies that are res istant to se pt oria leaf spot . 4 . ** Sanit ation **: Dis inf ect g ard ening tools and equipment used on the plant to prevent the spread of the disease . 5 . ** Water Management **: Avoid overhead water ing , which can increase the risk of disease . Water at the base of the plant to reduce the risk of water drop lets sp reading the disease . It 's important to note that the treatment should be applied as soon as possible to prevent the disease from sp reading . Regular monitoring of the plant for signs of disease is also cr uc ial for early inter vention ."

print(fix_split_words(text))