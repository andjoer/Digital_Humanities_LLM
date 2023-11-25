import re



dir_speech_closed_patterns = [r'»(.*?)«',r'›(.*?)‹'] 
dir_speech_open_patterns = [r'«(.*?)»',r'‹(.*?)›'] 

dir_speech_pattern = r'„(.*?)”'
dir_speech_asy_pattern = [('»','«'),('›','‹')]


def preprocess_text(text):
    text = re.sub('\r\n', ' ', text)
    text = re.sub('\n',' ', text)
    text = re.sub(r'(?<=[a-zA-ZÄÖÜäöüß])-\s*(?=[a-zA-ZÄÖÜäöüß])','', text)
    text = re.sub('[^A-Za-zÄÖÜäöüß«»‹›"„” -,.:;!?]',' ',text)
    return text

def get_asymmetric_speech(text):
    search_patterns = []
    for pattern in dir_speech_asy_pattern:
        try:
            if text.index(pattern[0]) < text.index(pattern[1]):
                search_patterns = dir_speech_closed_patterns
            else:
                search_patterns = dir_speech_open_patterns
        except:
            pass
        
    matches = []
    for pattern in search_patterns:
        matches += [(match.start(), match.end()) for match in re.finditer(pattern, text)]


    return matches

def get_symmetric_speech(text):
    matches = [match.start() for match in re.finditer('"',text)]
    return [(matches[i], matches[i+1]) for i in range(len(matches)-1)]

def search_speech(text):               # needed for nested patterns
    matches = []

    matches = [(match.start(), match.end()) for match in re.finditer(dir_speech_pattern, text)]

    if not matches:
        matches = get_asymmetric_speech(text)

    if not matches: 
        matches = get_symmetric_speech(text)

    return matches
