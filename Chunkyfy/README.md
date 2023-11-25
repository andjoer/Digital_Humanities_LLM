# App to extract text chunks
App to extract chunks of text with or without specific criteria specified by regex from large corpora.  

Work in progress

![alt text](https://github.com/andjoer/Digital_Humanities_LLM/blob/main/Chunkyfy/images/chunkyfy.png)


# Chunkyfy User Manual: Understanding the Basics

## Introduction

**Chunkyfy** is an application designed to extract specific segments or 'chunks' of text from large bodies of text (corpora) based on the occurrence of certain words or phrases, defined using regular expressions (regex). This manual aims to introduce the fundamental concepts and logic behind Chunkyfy to facilitate a basic understanding of its operation.

## Conceptual Overview

### What is a Chunk?

A 'chunk' in Chunkyfy is a segment of text extracted from a larger document. The boundaries of a chunk are defined by two main parameters:

1. **Occurrence of Specific Words/Phrases**: Identified through regex patterns.
2. **Character Spans**: The number of characters before and after the found word/phrase.

### Regular Expressions (Regex)

Regex is a powerful tool used in Chunkyfy to specify patterns of text. Users can define what to search for in the text corpus using these patterns.

### Frames and Regex Lines

- **Frames**: A frame in Chunkyfy is a container for one or more regex conditions. Each frame's conditions must be met for a chunk to be extracted.
- **Regex Lines**: Within each frame, users can add multiple lines, each specifying a different regex pattern and its associated conditions.

## Core Functionalities

### Adding and Configuring Frames

1. **Creating a Frame**: Users can create multiple frames, each with its own set of conditions.
2. **Frame Color**: Each frame is color-coded for easy distinction.
3. **Adding Regex Lines**: Within each frame, users can add regex lines to define the search criteria.

### Defining Search Criteria

For each regex pattern, users can specify:

1. **Occurrence Count**: Define the minimum and maximum number of times the pattern should appear in a chunk.
2. **Case Sensitivity**: Choose whether the search should be case-sensitive.
3. **Direct Speech Consideration**: Specify whether the pattern should be searched only within direct speech, outside of it, or regardless of it.
4. **Character before and after**: Define the number of characters before and after the regex match to include in the chunk.

- **Purpose**: These settings within a regex line define a focused search radius around the initial regex match that determined the chunk. They specify how much of the initially extracted chunk should be considered when checking for other regex patterns in the same line.
- **Functionality**:
  - **Characters Before**: Determines how many characters before the initially found regex match (that defined the chunk) are considered in this secondary validation.
  - **Characters After**: Specifies the number of characters after the initially found regex match to include in the validation check.
- **Usage Scenarios**:
  - **Non-zero Settings**: If 'Characters Before' and/or 'Characters After' are set to non-zero values, only the portion of the chunk within this defined radius is considered when checking for occurrences of the additional regex patterns in the line.
  - **Zero Settings**: If both are set to zero, the entire chunk extracted during the initial search is used for further regex checks in that line.

## Processing Text

### Initial Search Criteria

1. **Activating Regex Patterns**: For the app to use a regex pattern in the initial search, the checkbox next to the regex line must be checked. Only checked regex patterns are considered in this phase.
2. **Searching the Corpus**: Chunkyfy conducts the initial search through the text corpus using only the activated (checked) regex patterns in each frame.

### Subsequent Steps in Processing Text

1. **Chunk Extraction**: Upon finding a match with an active regex pattern, Chunkyfy extracts a chunk of text around it. This extraction is based on the defined character spans before and after the match.
2. **Further Validation**: Each extracted chunk is then evaluated to see if it meets all other conditions specified in the frame. These conditions include the occurrence count, case sensitivity, and direct speech settings.

### Acceptance of Chunks

A chunk is accepted and included in the final output if:

1. **Within a Frame**: It fulfills all the conditions of at least one frame.
2. **Across Frames**: It adheres to the requirements of any of the frames configured in the app.

## Opening a Folder/Files for a Corpus and Saving Selected Chunks

### Opening a Folder or Files

Chunkyfy allows users to load text data either from individual files or an entire folder. This flexibility is essential for working with various sizes and types of text corpora.

#### Steps to Load Text Data

1. **Open Folder/File Button**: Click the 'Open Folder/File' button on the main interface.
2. **Selecting Files or Folder**:
   - **Individual Files**: A dialog box will appear, allowing you to select multiple text files from your system.
   - **Entire Folder**: You can also choose to load all text files within a selected folder. This is useful for processing large datasets or entire document collections.


## Saving Selected Chunks

After processing your text data and extracting relevant chunks in Chunkyfy, you have the option to save these chunks for further analysis or record-keeping. Chunkyfy automatically saves these chunks in two formats: a pickled DataFrame (`.chk` file) for data integrity and a human-readable text file (`.txt` file) for easy viewing and analysis.

### Steps to Save Extracted Chunks

1. **Save Chunks Button**: After processing the text files, click the 'Save Chunks' button on the main interface.
   
2. **Automatic File Generation**:
   - **DataFrame File (.chk)**: The app automatically saves a pickled DataFrame of the chunks. This format preserves the data structure and is ideal for later data processing or analysis using Python.
   - **Text File (.txt)**: In addition to the DataFrame, a text file is created. This file is structured for readability, with each chunk preceded by its original filename and the character span from which it was extracted. 

3. **File Naming and Location**:
   - A dialog box will prompt you to choose a location and name for saving these files.
   - The `.chk` and `.txt` files will share the same base name you specify.

4. **File Structure of the Text File**:
   - The text file will list chunks in the order they were processed.
   - Each chunk will have a header indicating the filename from which it was extracted and the specific character span.
   - Chunks are separated by a line containing `***` for clear demarcation.


## Practical Use

Chunkyfy is useful in various scenarios like:

- **Text Analysis**: For linguistic research or data mining.
- **Keyword-Based Searching**: In large documents or databases.
- **Custom Data Extraction**: For specific research needs or information gathering.

