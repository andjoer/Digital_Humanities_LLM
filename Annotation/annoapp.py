from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import yaml
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QCheckBox, QVBoxLayout, QWidget, QTabWidget, QGridLayout, QHBoxLayout, QLabel, QMessageBox
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QComboBox, QLineEdit
import pandas as pd
import pickle

@dataclass
class Annotation:
    text: List[str]  # list of texts for multiple fields
    mode: str  # 'categorize' or 'answer'
    categories: List[str] = field(default_factory=list)
    spans: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)  # key: index of text, value: list of (start, end) spans
    answer: Optional[str] = None
    prompt: Optional[str] = None
    skip: bool = False



def load_config(yaml_file_path: str) -> dict:
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class AnnotationApp(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.annotations = [] 
        self.config = config
        self.init_ui()

        self.column_dropdown = QComboBox(self)
    
    def init_ui(self):
        self.textfields = []
        self.subtextfields = []
        self.textfield_to_label = {} 
        self.checkboxes = []
        self.selected_spans = {}
        self.current_idx = 0

        self.tabs = QTabWidget(self)

        # Create the file handling tab
        self.file_handling_tab = QWidget(self)
        self.tabs.addTab(self.file_handling_tab, "File Handling")

        # Create a layout for the file handling tab
        file_handling_layout = QVBoxLayout(self.file_handling_tab)

        # Button to open file
        self.open_file_btn = QPushButton('Open File', self)
        self.open_file_btn.clicked.connect(self.open_file)
        file_handling_layout.addWidget(self.open_file_btn)


        # Inside the init_ui method, under the file handling tab layout
        self.save_annotations_btn = QPushButton('Save Annotations', self)
        self.save_annotations_btn.clicked.connect(self.save_annotations)
        file_handling_layout.addWidget(self.save_annotations_btn)


        # Create the annotation tab
        self.annotation_tab = QWidget(self)
        self.tabs.addTab(self.annotation_tab, "Annotation")

        # Create a layout for the annotation tab and add all annotation-related widgets here
        annotation_layout = QVBoxLayout(self.annotation_tab) 

        # Text display area
        self.text_displays = []  # List to hold the QTextEdit widgets

        for col_name in (self.config['text_columns']):
            label = QLabel(col_name, self)  # Adding a label for clarity
            annotation_layout.addWidget(label)

            text_edit = QTextEdit(self)
            self.text_displays.append(text_edit)
            annotation_layout.addWidget(text_edit)
      
        # Textfield for the prompt
        if self.config.get('edit_prompt', False):
            label = QLabel('Prompt', self)  # Adding a label for clarity
            annotation_layout.addWidget(label)
            self.prompt_field = QTextEdit(self)
            annotation_layout.addWidget(self.prompt_field)
            self.prompt_field.setText(self.config.get('default_prompt', ''))


        # Textfield for the answer
        if self.config.get('answers', False):
            label = QLabel('Answer', self)  # Adding a label for clarity
            annotation_layout.addWidget(label)
            self.answer_field = QTextEdit(self)
            annotation_layout.addWidget(self.answer_field)
            self.answer_field.setText(self.config.get('default_answer', ''))


        if self.config['categories']:
            # Create a grid layout for categories and subcategories
            categories_grid_layout = QGridLayout()

            # Initialize dictionaries to store the relationship between textfields and their labels
            self.textfield_to_label = {}  # For main categories
            self.subtextfield_to_label = {}  # For subcategories
            self.sub_checkboxes = []
            self.main_cat_children ={}
            for row_idx, category_dict in enumerate(self.config['categories']):
                # Handle the main category
                category_type = category_dict.get('type', 'checkbox')  # Default to checkbox if not specified

                if category_type == "checkbox":
                    if 'subcategories' not in category_dict:
                        main_checkbox = QCheckBox(category_dict['name'], self)
                        self.checkboxes.append(main_checkbox)
                        categories_grid_layout.addWidget(main_checkbox, row_idx, 0)
                    else:
                        category_label = QLabel(category_dict['name'], self)
                        bold_font = QFont()
                        bold_font.setBold(False)
                        category_label.setFont(bold_font)
                        categories_grid_layout.addWidget(category_label, row_idx, 0)
                elif category_type == "text":
                    category_label = QLabel(category_dict['name'], self)
                    bold_font = QFont()
                    bold_font.setBold(False)
                    category_label.setFont(bold_font)
                    categories_grid_layout.addWidget(category_label, row_idx, 0)

                    textfield = QLineEdit(self)
                    categories_grid_layout.addWidget(textfield, row_idx, 1)
                    self.textfields.append(textfield)
                    self.textfield_to_label[textfield] = category_dict['name']  # store the relationship

                # Handle the subcategories, if any
                if 'subcategories' in category_dict:
                    # Create a widget to hold the subcategories
                    subcategories_widget = QWidget(self)
                    subcategory_layout = QHBoxLayout(subcategories_widget)
                    self.main_cat_children[category_dict['name']] = []
                    for subcategory in category_dict['subcategories']:
                        sub_checkbox = QCheckBox(subcategory['name'], self)
                        self.sub_checkboxes.append(sub_checkbox)
                        sub_checkbox.stateChanged.connect(lambda state, lbl=category_label: self.update_category_label(state, lbl))
                        subcategory_layout.addWidget(sub_checkbox)
                        
                        child = [sub_checkbox]
                        if subcategory.get('textfield'):
                            sub_textfield = QLineEdit(self)
                            subcategory_layout.addWidget(sub_textfield)
                            self.subtextfields.append(sub_textfield)
                            self.subtextfield_to_label[sub_textfield] = category_dict['name']  # Map sub-textfield to main category label
                            child.append(sub_textfield )
                        self.main_cat_children[category_dict['name']].append(child)   # child: [checkbox, textfield] or [checkbox]

                    categories_grid_layout.addWidget(subcategories_widget, row_idx, 1)


            # Add the categories grid layout to the main annotation layout
            annotation_layout.addLayout(categories_grid_layout)


            
            # Add buttons for span annotation if enabled
            if self.config.get('annotate_spans', False):
                span_btns_layout = QHBoxLayout()  # New horizontal layout for span buttons
                self.select_span_btn = QPushButton('Select Span', self)
                self.select_span_btn.clicked.connect(self.handle_span_selection)  
                span_btns_layout.addWidget(self.select_span_btn)
                
                self.deselect_span_btn = QPushButton('Deselect Span', self)
                self.deselect_span_btn.clicked.connect(self.handle_deselect_span)
                span_btns_layout.addWidget(self.deselect_span_btn)

                annotation_layout.addLayout(span_btns_layout)  # Add the horizontal layout to main layout

                self.annotate_other_span_btn = QPushButton('Annotate Other Span', self)
                self.annotate_other_span_btn.clicked.connect(self.handle_annotate_other_span)
                annotation_layout.addWidget(self.annotate_other_span_btn)

                self.selected_spans = {}

      
            
        
        # Common buttons
        next_skip_layout = QHBoxLayout()  # New horizontal layout for Next and Skip buttons
        self.next_btn = QPushButton('Next', self)
        self.next_btn.clicked.connect(self.handle_next)
        next_skip_layout.addWidget(self.next_btn)

        self.skip_btn = QPushButton('Skip', self)
        self.skip_btn.clicked.connect(self.handle_skip)
        next_skip_layout.addWidget(self.skip_btn)

        annotation_layout.addLayout(next_skip_layout)  # Add the horizontal layout to main layout

        backward_forward_layout = QHBoxLayout()  # New horizontal layout for Go Backward and Go Forward buttons
        self.backward_btn = QPushButton('Go Backward', self)
        self.backward_btn.clicked.connect(self.handle_backward)
        backward_forward_layout.addWidget(self.backward_btn)

        self.forward_btn = QPushButton('Go Forward', self)
        self.forward_btn.clicked.connect(self.handle_forward)
        backward_forward_layout.addWidget(self.forward_btn)

        annotation_layout.addLayout(backward_forward_layout)  # Add the horizontal layout to main layout

        self.setCentralWidget(self.tabs)

        self.setWindowTitle('Text Annotation App')
        self.show()

    
    def update_category_label(self, state, label):
        bold_font = QFont()
        bold_font.setBold(state == Qt.Checked)
        label.setFont(bold_font)

    def handle_span_selection(self):

        for active_text_display in self.text_displays:
            idx = self.text_displays.index(active_text_display)

            cursor = active_text_display.textCursor()
            start = cursor.selectionStart()
            end = cursor.selectionEnd()

            # Check if a valid span is selected
            if start != end:
               
                if idx not in self.selected_spans:
                    self.selected_spans[idx] = []
                self.selected_spans[idx].append((start, end))
        
                
                # Highlight the selected span
                highlight_format = QTextCharFormat()
                highlight_format.setBackground(QColor('yellow'))
                cursor.mergeCharFormat(highlight_format)
                active_text_display.mergeCurrentCharFormat(highlight_format)

                # Optionally, you can clear the selection after highlighting
                cursor.clearSelection()
                active_text_display.setTextCursor(cursor)

    def handle_deselect_span(self):
        for active_text_display in self.text_displays:
            idx = self.text_displays.index(active_text_display)
            cursor = active_text_display.textCursor()
            start = cursor.selectionStart()
            end = cursor.selectionEnd()

            # Check if a valid span is selected
            if start != end and idx in self.selected_spans.keys():
                new_spans = []
                for span_start, span_end in self.selected_spans[idx]:
                    # Check for intersection
                    if span_start < end and span_end > start:
                        # Remove highlighting from the intersecting portion
                        intersection_start = max(span_start, start)
                        intersection_end = min(span_end, end)
                        cursor.setPosition(intersection_start, QTextCursor.MoveAnchor)
                        cursor.setPosition(intersection_end, QTextCursor.KeepAnchor)
                        
                        # Explicitly set the background to transparent
                        transparent_format = QTextCharFormat()
                        transparent_format.setBackground(QColor(255, 255, 255, 0))  # RGBA: transparent white
                        cursor.mergeCharFormat(transparent_format)

                        # Update the spans list
                        if span_start < intersection_start:
                            new_spans.append((span_start, intersection_start))
                        if span_end > intersection_end:
                            new_spans.append((intersection_end, span_end))
                    else:
                        new_spans.append((span_start, span_end))
                self.selected_spans[idx] = new_spans

                # Clear the current selection
                cursor.clearSelection()
                active_text_display.setTextCursor(cursor)


    def store_current_annotation(self, insert = False):

        categories_selected = {checkbox.text():checkbox.isChecked() for checkbox in self.checkboxes}
        
        # Create a dictionary for main categories with text fields
        for textfield in self.textfields:
            categories_selected[self.textfield_to_label[textfield]] = textfield.text()
           

        # Assuming `self.subtextfields` contains the text fields for subcategories:
        for main_cat in self.main_cat_children.keys():
            print(main_cat)
            label_dict = {}
            checkboxes = self.main_cat_children[main_cat]
            for checkbox in checkboxes: 
                if len(checkbox) ==1:
                    label_dict[checkbox[0].text()] = checkbox[0].isChecked()

                elif len(checkbox) == 2:
                    if checkbox[0].isChecked():
                        label_dict[checkbox[0].text()] = checkbox[1].text()
                    else: 
                        label_dict[checkbox[0].text()] = False

            categories_selected[main_cat] = label_dict
        
        if self.config.get('edit_prompt', False):
            prompt = self.prompt_field.toPlainText()
        elif self.config.get('default_prompt', False):
            prompt = self.config['default_prompt']
        else: 
            prompt = None

        if self.config.get('answers', False):
            answer = self.answer_field.toPlainText()
        elif self.config.get('default_answer', False):  # there is no use for this edge case, but maybe someone might need it.
            answer = self.config['default_answer']
        else: 
            answer = None
        current_annotation = Annotation(
            text=[text_display.toPlainText() for text_display in self.text_displays],
            mode=self.config['mode'],
            categories=categories_selected,
            spans= self.selected_spans,
            answer=answer,
            prompt=prompt,
            skip=False
        )
        if not insert:
            self.annotations[self.current_idx] = current_annotation
        else: 
            self.annotations.insert(self.current_idx,current_annotation)
            self.current_idx += 1

    def get_next_idx(self):
        # Get the index for the next annotation
        for idx, annotation in enumerate(self.annotations):
            if annotation.mode is None:
                self.current_idx = idx
                return 
            
    def reset_selection(self):
        self.selected_spans = {}

        for text_display in self.text_displays:
            # Create a cursor for the QTextEdit
            cursor = text_display.textCursor()

            # Select the entire text
            cursor.select(QTextCursor.Document)

            # Create a default QTextCharFormat object
            transparent_format = QTextCharFormat()
            transparent_format.setBackground(QColor(255, 255, 255, 0))  # RGBA: transparent white

            # Apply the default format to the selected text
            cursor.mergeCharFormat(transparent_format)

            # Set the cursor back to the QTextEdit
            text_display.setTextCursor(cursor)
            # Clear the selection
            
            cursor.clearSelection()
            text_display.setTextCursor(cursor)

         # Reset checkboxes
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)
        # Clear subcategory checkboxes
        for main_cat in self.main_cat_children.keys():
            for checkbox in self.main_cat_children[main_cat]:
                checkbox[0].setChecked(False)
                # Clear subcategory text fields
                if len(checkbox) == 2:
                    checkbox[1].clear()

        # Clear main category text fields
        for textfield in self.textfields:
            textfield.clear()


        # Reset prompt field
        if self.config.get('edit_prompt', False):
            self.prompt_field.setText(self.config.get('default_prompt', ''))
        if self.config.get('answers', False):
            self.answer_field.setText(self.config.get('default_answer', ''))

    def handle_next(self):
        # Store the current annotation
        self.store_current_annotation()
        for anno in self.annotations:
            print(anno.categories)

            print(anno.prompt)
        print('###')
        
        
        previous_idx = self.current_idx
        self.get_next_idx()

        if previous_idx == self.current_idx:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)  # Set the icon to Information (can also use Warning, Critical, etc.)
            msg.setText("End of dataset.")
            msg.setWindowTitle("End of dataset")
            msg.setStandardButtons(QMessageBox.Ok)  # You can add more buttons if needed
            msg.exec_()

        else:
            self.reset_selection()
            for i, text_column in enumerate(self.config['text_columns']):
                self.text_displays[i].setText(self.annotations[self.current_idx].text[i])



    def handle_annotate_other_span(self):
       self.store_current_annotation(insert=True)
       self.reset_selection()


    def handle_skip(self):
        current_annotation = Annotation(
            text=[text_display.toPlainText() for text_display in self.text_displays],
            mode=self.config['mode'],
            categories=[],
            spans={},
            answer=None,
            prompt=None,
            skip=True
        )
        self.annotations.append(current_annotation)
        self.reset_selection()
        self.get_next_idx()
        
        self.text_display.setText(self.annotations[self.current_idx].text)


    def open_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Pandas Pickle Files (*.pkl);;Annotation App Files (*.ano);;All Files (*)", options=options)
        
        if file_name:
            # Check file extension to determine how to read the file
            if file_name.endswith('.ano'):
                self.annotations = pickle.load(open(file_name, "rb"))
                self.current_idx = 0
                self.update_display()
            else: 
                self.annotations = []
                if file_name.endswith('.csv'):
                    self.dataframe = pd.read_csv(file_name)
                elif file_name.endswith('.pkl'):
                    self.dataframe = pd.read_pickle(file_name)

                self.current_idx = 0

                print("Dropdown items:", self.column_dropdown.count())
                for i in range(self.column_dropdown.count()):
                    print(self.column_dropdown.itemText(i))
                texts = [[row[col] for col in self.config['text_columns']] for _, row in self.dataframe.iterrows()]
                self.annotations = [Annotation(text=text_list, mode=None, categories=[], spans={}, answer=None, prompt=None, skip=False) for text_list in texts]
                for i, text_column in enumerate(self.config['text_columns']):
                    self.text_displays[i].setText(self.annotations[self.current_idx].text[i])
    
    def handle_backward(self):

        if not self.selected_spans:                                        # if this is an already existent annotation
            self.selected_spans = self.annotations[self.current_idx].spans
        self.store_current_annotation()
        if self.current_idx > 0:
            self.current_idx -= 1
         
            self.update_display()

    def handle_forward(self):
        if not self.selected_spans:
            self.selected_spans = self.annotations[self.current_idx].spans
        self.store_current_annotation()
        if self.current_idx < len(self.annotations) - 1:
            self.current_idx += 1
        
            self.update_display()

    def update_display(self):
        # Update text display
        for i, text_column in enumerate(self.config['text_columns']):
            self.text_displays[i].setText(self.annotations[self.current_idx].text[i])

        # Reset selection and checkboxes
        self.reset_selection()


        # Update checkboxes and text fields
        for category_name in self.annotations[self.current_idx].categories.keys():
            category_value = self.annotations[self.current_idx].categories[category_name]
            if isinstance(category_value,bool) and category_value == True:
                for checkbox in self.checkboxes:
                    if checkbox.text() == category_name:
                        checkbox.setChecked(True)

            elif isinstance(category_value,str):
                for textfield in self.textfields:
                    if self.textfield_to_label[textfield] == category_name:
                        textfield.setText(category_value)

            elif isinstance(category_value,dict):
                checkboxes = self.main_cat_children[category_name]
                for checkbox in checkboxes: 
                    if checkbox[0].text() in category_value.keys():
                        if  category_value[checkbox[0].text()] != False:
                            checkbox[0].setChecked(True)

                            if len(checkbox) == 2:
                                checkbox[1].setText(category_value[checkbox[0].text()])


        # Update prompt display
        if self.config.get('answers', False):
            self.prompt_field.setText(self.annotations[self.current_idx].prompt)
        if self.config.get('answers', False): 
            self.answer_field.setText(self.annotations[self.current_idx].answer)
        # Update spans
    
        for text_display in self.text_displays:
            idx = self.text_displays.index(text_display)
            if idx in self.annotations[self.current_idx].spans.keys():
                cursor = text_display.textCursor()
                highlight_format = QTextCharFormat()
                highlight_format.setBackground(QColor('yellow'))
                for start, end in self.annotations[self.current_idx].spans[idx]:
                    cursor.setPosition(start, QTextCursor.MoveAnchor)
                    cursor.setPosition(end, QTextCursor.KeepAnchor)
                    cursor.mergeCharFormat(highlight_format)

        
    def save_annotations(self):
        if not self.selected_spans:
            self.selected_spans = self.annotations[self.current_idx].spans
        self.store_current_annotation()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "Annotations Files (*.ano);;All Files (*)", options=options)
        
        if file_name:
            # Ensure the file has the correct extension
            if not file_name.endswith('.ano'):
                file_name += '.ano'
            
            # Save the annotations list using pickle
            with open(file_name, 'wb') as file:
                pickle.dump(self.annotations, file)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    config = load_config('config.yml')
    ex = AnnotationApp(config)
    sys.exit(app.exec_())
