import sys
from PyQt5.QtWidgets import (QWidget, QLabel, QLineEdit, QSpinBox, QPushButton, QVBoxLayout, QHBoxLayout, QApplication, QFrame, QFileDialog, QMessageBox,QScrollArea, QCheckBox, QComboBox)
from PyQt5.QtGui import QPalette, QColor

import os
import pandas as pd
import re
from FileHandling import open_file
from typing import List, Tuple

from TextHandling import search_speech


class RegexFrame(QFrame):
    color_index = 0  # Static variable to keep track of the next color index
    colors = ["lightblue", "lightgreen", "lightyellow", "lightpink", "lightgrey"]  # List of colors
    def __init__(self, remove_callback):
        super().__init__()
        self.remove_callback = remove_callback
        self.regex_fields = []  # List to store RegexField objects
        self.initUI()

    def initUI(self):
        # Set the background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        color = QColor(RegexFrame.colors[RegexFrame.color_index])
        palette.setColor(QPalette.Window, color)
        self.setPalette(palette)

        # Update the color index for the next frame
        RegexFrame.color_index = (RegexFrame.color_index + 1) % len(RegexFrame.colors)

        self.layout = QVBoxLayout(self)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)

        # Layout for the add and remove buttons
        self.buttons_layout = QHBoxLayout()
        
       
        # Button to add regex condition
        self.add_regex_button = QPushButton('Add Regex Field', self)
        self.add_regex_button.clicked.connect(self.add_regex_field)
        self.buttons_layout.addWidget(self.add_regex_button)
        
        # Button to remove the entire frame
        self.remove_frame_button = QPushButton('Remove Frame', self)
        self.remove_frame_button.clicked.connect(self.remove_self)
        self.buttons_layout.addWidget(self.remove_frame_button)
                # Layout for character count settings specific to this frame
        char_count_layout = QHBoxLayout()


        self.layout.addLayout(char_count_layout)

        self.layout.addLayout(self.buttons_layout)

        # Placeholder for regex fields
        self.regex_fields_layout = QVBoxLayout()
        self.layout.addLayout(self.regex_fields_layout)

        self.add_regex_field()

    def add_regex_field(self):
        regex_field = RegexField(self.remove_regex_field)
        self.regex_fields_layout.addWidget(regex_field)
        self.regex_fields.append(regex_field)  # Add the new RegexField to the list

    def remove_self(self):
        self.remove_callback(self)

    def remove_regex_field(self, regex_field):
        self.regex_fields.remove(regex_field)  # Remove the RegexField from the list
        regex_field.deleteLater()


class RegexField(QWidget):
    def __init__(self, remove_callback):
        super().__init__()
        self.remove_callback = remove_callback  # Callback to handle removal
        self.initUI()

    def initUI(self):
        self.layout = QHBoxLayout(self)

        # Checkbox
        self.search_checkbox = QCheckBox(self)
        self.search_checkbox.setChecked(True)
        self.layout.addWidget(self.search_checkbox)

        # Regex input field
        self.regex_input = QLineEdit(self)
        self.layout.addWidget(QLabel("Regex"))
        self.layout.addWidget(self.regex_input)

        # Min and Max occurrence spin boxes
        self.min_occurrence = QSpinBox(self)
        self.max_occurrence = QSpinBox(self)
        self.min_occurrence.setMinimum(-100)
        self.max_occurrence.setMinimum(-100)
        self.layout.addWidget(QLabel("MIN"))
        self.layout.addWidget(self.min_occurrence)
        self.layout.addWidget(QLabel("MAX"))
        self.layout.addWidget(self.max_occurrence)


        # Checkbox
        self.layout.addWidget(QLabel("CaseSensitive"))
        self.case_checkbox = QCheckBox(self)
        self.case_checkbox.setChecked(True)
        self.layout.addWidget(self.case_checkbox)

        # Character count fields for the radius around the word
        self.chars_before = QSpinBox(self)
        self.chars_after = QSpinBox(self)
        self.layout.addWidget(QLabel("Chars Before"))
        self.layout.addWidget(self.chars_before)
        self.layout.addWidget(QLabel("Chars After"))
        self.layout.addWidget(self.chars_after)

        # Dropdown for direct speech
        self.direct_speech_dropdown = QComboBox(self)
        self.direct_speech_dropdown.addItems(["-", "Yes", "No"])
        self.layout.addWidget(QLabel("In Direct Speech"))
        self.layout.addWidget(self.direct_speech_dropdown)

        # Remove button
        self.remove_button = QPushButton('-', self)
        self.remove_button.clicked.connect(self.remove_self)
        self.layout.addWidget(self.remove_button)

    def remove_self(self):
        self.remove_callback(self)  # Invoke the callback
        self.setParent(None)

    def remove_regex_field(self, regex_field):
        self.regex_fields.remove(regex_field)  # Remove the RegexField from the list
        regex_field.deleteLater()

class ChunkyfyUI(QWidget):
    def __init__(self):
        super().__init__()
        self.file_paths = []
        self.regex_frames = []  # List to store RegexFrame objects
        self.initUI()

    def initUI(self):
        self.scrollArea = QScrollArea(self)  # Create a Scroll Area
        self.scrollArea.setWidgetResizable(True)  # Make the scroll area resizable

        # Main widget that will contain your layout
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)  # Set the layout on the main widget
        #self.main_layout = QVBoxLayout(self)
        
        # Button to add new regex frame
        self.add_frame_button = QPushButton('Add Frame', self)

        self.save_chunks_button = QPushButton('Save Chunks', self)
        self.save_chunks_button.clicked.connect(self.save_chunks)
        self.main_layout.addWidget(self.save_chunks_button)

        self.process_button = QPushButton('Process files', self)
        self.process_button.clicked.connect(self.process_files)
        self.main_layout.addWidget(self.process_button)

        self.open_button = QPushButton('Open Folder/File', self)
        self.open_button.clicked.connect(self.open_file_dialog)
        self.main_layout.addWidget(self.open_button)

        self.save_config_button = QPushButton('Save Configuration', self)
        self.save_config_button.clicked.connect(self.save_config)
        self.main_layout.addWidget(self.save_config_button)

        self.load_config_button = QPushButton('Load Configuration', self)
        self.load_config_button.clicked.connect(self.load_config)
        self.main_layout.addWidget(self.load_config_button)

        self.add_frame_button.clicked.connect(self.add_regex_frame)
        self.main_layout.addWidget(self.add_frame_button)
        global_char_count_layout = QHBoxLayout()
        self.global_chars_before = QSpinBox(self)
        self.global_chars_after = QSpinBox(self)
        self.global_overlap = QSpinBox(self)
        global_char_count_layout.addWidget(QLabel("Global Characters Before"))
        global_char_count_layout.addWidget(self.global_chars_before)
        global_char_count_layout.addWidget(QLabel("Global Characters After"))
        global_char_count_layout.addWidget(self.global_chars_after)
        global_char_count_layout.addWidget(QLabel("Overlap"))
        global_char_count_layout.addWidget(self.global_overlap)

        self.global_chars_before.setValue(100)
        self.global_chars_after.setValue(100)


        self.main_layout.addLayout(global_char_count_layout)
        # Placeholder for regex frames
        self.regex_frames_layout = QVBoxLayout()
        self.main_layout.addLayout(self.regex_frames_layout)
        self.add_regex_frame()
         # Setting the main widget as the scroll area's widget
        self.scrollArea.setWidget(self.main_widget)

        # Main layout for the outer window
        outer_layout = QVBoxLayout(self)  # This is the layout for the ChunkyfyUI
        outer_layout.addWidget(self.scrollArea)

        self.setLayout(outer_layout)  # Set the outer layout to the window
    def add_regex_frame(self):
        regex_frame = RegexFrame(self.remove_regex_frame)
        self.regex_frames_layout.addWidget(regex_frame)
        self.regex_frames.append(regex_frame)  # Add the new RegexFrame to the list

    def remove_regex_frame(self, frame):
        # Remove frame from layout and delete the object
        self.regex_frames_layout.removeWidget(frame)
        self.regex_frames.remove(frame)  # Remove the RegexFrame from the list
        frame.deleteLater()

    def open_file_dialog(self):
        choice = self.ask_user_choice()
        if choice == "files":
            self.select_files()
        elif choice == "folder":
            self.select_folder()

    def ask_user_choice(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Select Files or Folder")
        msg_box.setText("Do you want to select files or a folder?")
        file_button = msg_box.addButton("Files", QMessageBox.AcceptRole)
        folder_button = msg_box.addButton("Folder", QMessageBox.RejectRole)
        msg_box.exec_()

        if msg_box.clickedButton() == file_button:
            return "files"
        elif msg_box.clickedButton() == folder_button:
            return "folder"
        return None

    def select_files(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "All Files (*);;Text Files (*.txt)", options=options)
        if files:
            self.file_paths = files
            print(self.file_paths)  # For demonstration purposes

    def select_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if folder:
            self.file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            print(self.file_paths)  # For demonstration purposes


    def process_files(self):

        data = {'filename': [], 'fulltext': []}

        for file_path in self.file_paths:
            content = open_file(file_path)
            data['filename'].append(os.path.basename(file_path))
            data['fulltext'].append(content)

        data['chunks'] = []

        for fulltext in data['fulltext']:
            text_processor = TextProcessor(fulltext, self.regex_frames, self.global_chars_before.value(), self.global_chars_after.value())
            data['chunks'].append(text_processor.process_text())

        self.chunks_df =  pd.DataFrame(data)
        print(self.chunks_df.head())

            
    def save_chunks(self):
        # Open a file dialog to select the save location
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, 
                                                    "Save Chunks", 
                                                    "", 
                                                    "Chunks Files (*.chk)", 
                                                    options=options)
        if fileName:
            if not fileName.endswith('.chk'):
                fileName += '.chk'

            # Save the DataFrame to a pickle file
            self.chunks_df.to_pickle(fileName)

        
        with open(fileName[:-4]+'.txt', 'w') as file:
            for _, row in self.chunks_df.iterrows():

                # Iterating through the list of tuples in 'chunks'
                for chunk in row['chunks']:
                    file.write(f"{row['filename']}\n")
                    # Writing the second element of the tuple and the string (paragraph)
                    file.write(f"{chunk[1]}\n{chunk[0]}\n")

                    # Separating paragraphs
                    file.write("\n***\n")


    def save_config(self):
        pass

    def load_config(self):
        pass
    

class TextProcessor:
    def __init__(self, text_corpus: str, regex_frames: List[RegexFrame], global_chars_before: int, global_chars_after: int):
        self.text_corpus = text_corpus
        self.regex_frames = regex_frames
        self.chars_before = global_chars_before
        self.chars_after = global_chars_after
    
        self.direct_speech_relevant = False
        for frame in regex_frames:
            for field in frame.regex_fields:
                if field.direct_speech_dropdown.currentText() != "-":
                    self.direct_speech_relevant = True
                    break

        if self.direct_speech_relevant:
            self.direct_speech = search_speech(text_corpus)


    def get_unique_regexes_from_frames(self) -> List[str]:

        regex_dict = {}
        for frame in self.regex_frames:
            for field in frame.regex_fields:
                if field.search_checkbox.isChecked() and field.min_occurrence.value() > 0: 
                    if field.case_checkbox.isChecked() not in regex_dict:
                        regex_dict[field.case_checkbox.isChecked()] = {}
                    if field.direct_speech_dropdown.currentText() not in regex_dict[field.case_checkbox.isChecked()]:
                        regex_dict[field.case_checkbox.isChecked()][field.direct_speech_dropdown.currentText()] = set()
       
                    expressions = field.regex_input.text().split('|')
                    regex_dict[field.case_checkbox.isChecked()][field.direct_speech_dropdown.currentText()].update(expressions)

        return regex_dict

    def find_all_occurrences(self, text: str, regex_dict: dict) -> List[Tuple[int, int]]:
        matches = []
        for case,regex_speech_dict in regex_dict.items():
            if case:
                for speech, regex in regex_speech_dict.items():

                    merched_regex = '|'.join(regex)
                    if speech == 'Yes':
                        matches += [(match.start(), match.end()) for match in re.finditer(merched_regex, text) if self.in_direct_speech(match.start(),match.end())]
                    elif speech == 'No':
                        matches += [(match.start(), match.end()) for match in re.finditer(merched_regex, text) if not self.in_direct_speech(match.start(),match.end())]
                    else:
                        matches += [(match.start(), match.end()) for match in re.finditer(merched_regex, text)]
            else:
                for speech, regex in regex_speech_dict.items():
                    merched_regex = '|'.join(regex)
                    if speech == 'Yes':
                        matches += [(match.start(), match.end()) for match in re.finditer(merched_regex, text, re.IGNORECASE) if self.in_direct_speech(match.start(),match.end())]
                    elif speech == 'No':
                        matches += [(match.start(), match.end()) for match in re.finditer(merched_regex, text, re.IGNORECASE) if not self.in_direct_speech(match.start(),match.end())]
                    else:
                        matches += [(match.start(), match.end()) for match in re.finditer(merched_regex, text, re.IGNORECASE)]
        return matches

    def extract_chunk(self, text: str, start: int, end: int) -> str:
        return text[max(0, start - self.chars_before):min(end + self.chars_after, len(text))]

    def is_valid_chunk(self, chunk: Tuple[str,Tuple[int,int]]) -> bool:
        for frame in self.regex_frames:
            if self.is_chunk_valid_for_frame(chunk, frame.regex_fields):
                return True
        return False

    def in_direct_speech(self,start,end):

        return any(start >= ds_start and end <= ds_end for ds_start, ds_end in self.direct_speech)
    
    def is_in_direct_speech(self, start, end,regex_field):
        # Check if the text chunk is within any direct speech span
        in_direct_speech = any(start >= ds_start and end <= ds_end for ds_start, ds_end in self.direct_speech)

        if regex_field.direct_speech_dropdown.currentText() == "Yes":
            return in_direct_speech
        elif regex_field.direct_speech_dropdown.currentText() == "No":
            return not in_direct_speech
        else:  # self.direct_speech_allowed == "-"
            return True
        
    def relevant_chunk(self, chunk: Tuple[str,Tuple[int,int]], regex_field: RegexField):

        start,end = chunk[1]
        text = chunk[0]

        if regex_field.chars_before.value() != 0 or regex_field.chars_after.value() != 0: 


            new_start = start - regex_field.chars_before.value()  #new start in absolute text position
            #new_end = end + regex_field.chars_after.value()  # new end in absolute text position (we do not know the text)

            # Calculate the relative positions of the new chunk
            new_relative_start = self.chars_before - regex_field.chars_before.value() 
            new_relative_end = end - start + regex_field.chars_after.value() + new_relative_start

            smaller_chunk = text[new_relative_start:new_relative_end]

        else:

            new_start = start - self.chars_before 
            
            new_relative_start = self.chars_before 
            new_relative_end = self.chars_before + end-start + self.chars_after
            smaller_chunk = text

        # Determine the action based on direct speech dropdown
        direct_speech_action = regex_field.direct_speech_dropdown.currentText()

        if direct_speech_action != "-":
            
            # Initialize an empty list to hold the processed spans
            processed_spans = []

            current_index = 0
            for ds_start, ds_end in self.direct_speech:
                # Adjust direct speech indices relative to the smaller chunk
                ds_start_relative = ds_start - new_start
                ds_end_relative = ds_end - new_start

                if direct_speech_action == "Yes":
                    # Add only the direct speech parts
                    if ds_end_relative >= 0:
                        processed_spans.append(smaller_chunk[max(0,ds_start_relative):ds_end_relative])
                    elif ds_start_relative >=0:
                        processed_spans.append(smaller_chunk[ds_start_relative:ds_end_relative])
                        

                elif direct_speech_action == "No":
                    # Skip the direct speech parts

                    if ds_start_relative > 0 and ds_start_relative < len(smaller_chunk): 
                      
                        processed_spans.append(smaller_chunk[current_index:ds_start_relative])
                    elif ds_start_relative > 0 and ds_start_relative > len(smaller_chunk):
                        processed_spans.append(smaller_chunk[current_index:ds_start_relative])

              
                current_index = max(current_index, ds_end_relative)
                if ds_end_relative > len(smaller_chunk):

                    break
           


            if direct_speech_action == "No" and current_index < len(smaller_chunk):
                processed_spans.append(smaller_chunk[current_index:])

            smaller_chunk = " ".join(processed_spans)

        return smaller_chunk
    

    def is_chunk_valid_for_frame(self, chunk: Tuple[str,Tuple[int,int]], regex_fields: RegexField) -> bool:

        for field in regex_fields:
                
            regex = field.regex_input.text()

            min_occurrence = field.min_occurrence.value()
            max_occurrence = field.max_occurrence.value()
            relevant_chunk = self.relevant_chunk(chunk, field)                      # allows for different search radius and direct/indirect speech selection

            if field.case_checkbox.isChecked():
                occurrences = len(re.findall(regex, relevant_chunk))
            else:
                occurrences = len(re.findall(regex, relevant_chunk, re.IGNORECASE))

            if (min_occurrence > occurrences or (occurrences > max_occurrence and not (max_occurrence == 0 and min_occurrence > 0))):  # !!! not optimal logic, maybe better usable

                return False
        return True

    def process_text(self) -> list:
        all_matches = {}
        regex_dict = self.get_unique_regexes_from_frames()

        all_matches = self.find_all_occurrences(self.text_corpus, regex_dict)

        chunks = []
        for (start,end) in all_matches:
   
            chunk = self.extract_chunk(self.text_corpus, start, end)
            chunks.append((chunk,(start,end)))

   
        valid_chunks = []
        for chunk in chunks:
            if self.is_valid_chunk(chunk):
                valid_chunks.append(chunk)

        return valid_chunks

        
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ChunkyfyUI()
    ex.show()
    sys.exit(app.exec_())