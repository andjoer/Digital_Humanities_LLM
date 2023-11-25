import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
import chardet    
from joblib import Parallel, delayed


def open_pdf(path):
    
    # creating a pdf file object
    pdfFileObj = open(path, 'rb')
    
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    
    
    # creating a page object

    text = ''

    for page in pdfReader.pages:
        pageObj = page
    
        # extracting text from page
        text += pageObj.extract_text() + '\n'
    
    # closing the pdf file object
    pdfFileObj.close()

    return text

def chapter_to_str(chapter):         # from https://andrew-muller.medium.com/getting-text-from-epub-files-in-python-fbfe5df5c2da
        soup = BeautifulSoup(chapter.get_body_content(), 'html.parser')
        text = [para.get_text() for para in soup.find_all('p')]
        return ' '.join(text)


def open_epub(path):
    
    book = epub.read_epub(path)

    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    
    text = ''

    for item in items:
        text += '\n' + chapter_to_str(item)

    return text

def open_rtf(path):
    with open(path) as infile:
        content = infile.read()
        text = rtf_to_text(content)
    return text
      
def open_txt(path):
    with open(path, 'rb') as f:
        rawdata = f.read()
        result = chardet.detect(rawdata)
        text= rawdata.decode(result['encoding'])

    return text


def open_file(path):
     
    format = path.split('.')[-1]
    try:
        if format == 'txt':
            text = open_txt(path)

        elif format == 'pdf':
            text = open_pdf(path)
        elif format == 'epub':
            text = open_epub(path)
        elif format == 'rtf':
            text = open_rtf(path)
        else: 
            text = ''

        return text
    except: 
        return ''
