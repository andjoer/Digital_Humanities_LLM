# Text Annotation App
Simple App to annotate texts and text spans. 

Work in progress

![alt text](Annotation/images/Annapp_gui.png)

## Configuring the App

The Text Annotation App is designed to be highly customizable through a configuration file in YAML format. This file dictates the structure and features available within the app, allowing users to tailor the annotation environment to their specific needs. Below is an overview of how various aspects of the app can be configured:

### Text and Image Columns:
- `text_columns`: Lists the columns from a dataset that contain the text to be annotated.
- `image_columns`: Lists the columns that contain paths to image data for annotation.

### Categories and Subcategories:
- `categories`: A list where each item represents a category for annotation. Each category can have a `name` and optional `subcategories`, each with its own `name`.
- Subcategories can be simple check boxes or have an additional text field (`textfield: true`) for more detailed annotations.
- Categories without subcategories will be represented as a single checkbox.
- The `type` attribute can be set to `checkbox` or `text`. `checkbox` will render a checkbox in the UI, and `text` will render a text field.

### Annotation Types:
- `annotate_spans`: When `true`, it enables the functionality to select and annotate specific spans of text within the provided texts.

### Prompt and Answer:
- `edit_prompt`: If `true`, it allows the user to edit the prompt in the UI.
- `answers`: If `true`, it allows the user to provide an answer or annotation in a text field in the UI.
- `default_prompt`: Sets a default prompt text in the UI.
- `default_answer`: Sets a default answer text in the UI.


## Annotation Process Using the Text Annotation App

### File Handling Tab:

This is where you begin your annotation process by loading the data you wish to work with:

1. **Open a File for Text Annotation**:
   - Click the ‘Open File’ button to load your data for text annotation.
   - Choose a CSV, Pandas Pickle, or Annotation App file that contains the text to be annotated.
   - With `text_columns` set in your YAML configuration, the app displays the text in the main annotation area for you to annotate.

2. **Load Images for Annotation**:
   - For image annotations, you can:
     - **Load from CSV**: Ensure `image_columns` in the YAML config is set with the columns containing image paths. When a CSV is opened, images from those paths are displayed.
     - **Load Image Folder**: Click the ‘Load Image Folder’ button to select and load images from a folder.

3. **Load Existing Annotations**:
   - To continue annotating from a previous session, you can load an existing annotation file:
     - Click the ‘Open File’ button and select an ".ano" file, which is a saved annotation file from this Annotation App.
     - The app will load the annotations allowing you to pick up right where you left off, whether to review, continue, or edit previous annotations.

### Annotation Tab:

After loading the data in the 'File Handling' tab, move to the 'Annotation' tab to start the annotation process:

### Annotating Spans of Text:

1. **Select Span**:
   - To annotate specific text spans, click and drag to highlight the desired section in the text display area.
   - Click the ‘Select Span’ button to mark the highlighted text. The background color of the text usually changes to indicate it has been selected.
   - You can add more selections by highlighting other parts of the text and clicking ‘Select Span’ again. Each new selection will be added to the existing set of annotations.

2. **Deselect Span**:
   - If you need to remove a highlighted span, re-highlight it and click the ‘Deselect Span’ button.
   - If multiple spans are selected and you wish to deselect only one, make sure to highlight only that span before clicking ‘Deselect Span’.

### Annotating Using Categories:

When annotating using categories, you have several options depending on how the categories are configured in the YAML file:

1. **Checkboxes for Categories and Subcategories**: For categories that are represented by checkboxes (including those with subcategories), you simply click the checkbox to select or deselect it. This indicates whether the specific category is relevant to the selected text or the entire text entry if no specific span is selected.

2. **Text Fields for Categories**: Some categories may be configured as "text" in the YAML file. For these categories, instead of checkboxes, you will see text fields where you can type in your annotations. This is particularly useful for adding more nuanced or detailed information that cannot be captured by a simple checkbox.

3. **Subcategories with Text Fields**: For subcategories that include an optional text field (`textfield: true` in the YAML configuration), you can first check the box to select the subcategory and then provide additional details in the text field that appears next to it.

4. **Combining Checkboxes and Text Fields**: If a category has both checkboxes and text fields, you can use them in conjunction to provide a more comprehensive annotation. For instance, you might select a checkbox to indicate a broader category and then use the text field to specify particular aspects of that category.

### Using the Annotation Controls:
6. **Annotate Other Span**: If you need to annotate another span of text in the same entry, you can use the ‘Annotate Other Span’ button. This allows you to store the current annotation and immediately proceed to select and annotate a new span without moving to the next text entry.
7. **Next**: After completing the annotations for a text entry, click the ‘Next’ button to save your work and move on to the next text entry in the dataset.
8. **Skip**: If you encounter a text entry that you cannot or do not want to annotate, you can use the ‘Skip’ button. This marks the current entry as skipped and moves to the next one.
9. **Go Backward**: If you need to revisit the previous text entry, you can click the ‘Go Backward’ button. This takes you back to the last annotated entry, allowing you to review or modify your annotations.
10. **Go Forward**: Similarly, if you have navigated backwards, you can use the ‘Go Forward’ button to return to where you left off in the annotation sequence.

### Finishing Up:
11. **Save Annotations**: Once you have completed your annotations, you can save your work by clicking the ‘Save Annotations’ button. This allows you to export the annotated data for future use or analysis.

Throughout the annotation process, the app will store your progress internally each time you click ‘Next’, ‘Skip’, or switch entries using ‘Go Backward’ or ‘Go Forward’. It's important to manually save regularly on your harddrive to ensure that no data is lost, especially after annotating a significant portion of the dataset.