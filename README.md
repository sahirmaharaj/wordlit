# Wordlit.net

<a href= "http://wordlit.net"> Wordlit.net </a> is aimed at demystifying the decisions and behaviors of algorithms in Natural Language Processing (NLP). It visualizes the relationships and entities extracted from text, offering insights into how NLP algorithms interpret and process language. The application is helpful for NLP researchers, data scientists, and enthusiasts keen on understanding the workings of computational linguistics. Wordlit.net currently supports input for file types of PDF, Word, and TXT.

![Wordlit-Home](https://github.com/sahirmaharaj/wordlit/assets/157965615/85dfdd74-e59d-4047-a1e3-3aa61714aaee)

## Key Features
- Entity Extraction: Leverages spaCy's NLP capabilities to identify entities in the text.
- Knowledge Graph Construction: Builds a graph using NetworkX, linking entities based on their relationships.
- Interactive Visualization: Utilizes Plotly and Streamlit for dynamic graph visualization.
- Customizable Graph Parameters: Offers options to adjust layout spacing, color scheme, node size, and more.
- Graph Analytics: Provides statistics like node and edge counts, graph density and centrality measures.
- Text Analytics: Calculates various text statistics such as token counts, sentence lengths, and unique tokens.

# Installation

To use this tool, you need to install the following dependencies:

`pip install spacy networkx transformers streamlit plotly matplotlib pandas`

![Install dependencies](https://sahirmaharaj.com/wordlit/Install%20Dependencies.png)

Don't forget to download the spaCy language model:

`python -m spacy download en_core_web_sm`

![Download the spaCy language model](https://sahirmaharaj.com/wordlit/Wordlit-Install.png)

# Usage

**1. Start the Streamlit App**: Run the app using Streamlit

`streamlit run wordlit.py`

![Run the app using Streamlit](https://sahirmaharaj.com/wordlit/Streamlit.png)

**2. Input Text**: You can input the text by uploading a file, inputting a website URL or pasting it directly into the text area provided.

![Input Text](https://sahirmaharaj.com/wordlit/Wordlit-Text-Input.png)

**3. Customize Graph**: Adjust the graph parameters like layout spacing, node size, and color scheme using the sidebar options.

![Customize Graph](https://sahirmaharaj.com/wordlit/Wordlit-Customize.png)

**4. Generate Graph**: Select 'Generate Graph' to visualize the knowledge graph based on your text.

![Generate Graph](https://sahirmaharaj.com/wordlit/Wordlit-Generate.png)

![Generate Graph](https://sahirmaharaj.com/wordlit/Wordlit-Generate_FullScreen.png)

**5. Explore Graph Analytics**: View various statistics and metrics related to the generated graph and the input text.

![Explore Graph Analytics](https://sahirmaharaj.com/wordlit/Wordlit-Analytics.png)

# Examples

Below is an example of a knowledge graph generated from a file. The nodes represent entities, and edges represent their relationships. Each node's size corresponds to its connection degree, and colors vary based on the selected color scheme.

https://github.com/sahirmaharaj/sahirmaharaj/assets/157965615/9bd5386b-e730-4c2b-8388-35d8babb9907

An example of a knowledge graph generated from text.

https://github.com/sahirmaharaj/sahirmaharaj/assets/157965615/bcf92fe0-bf40-43f5-b217-89abf4448ddf

An example of a knowledge graph generated from a website URL.

https://github.com/sahirmaharaj/sahirmaharaj/assets/157965615/79800d57-eb5b-41c8-801c-a1ca7255f415

# Tech Stack


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/800px-Python-logo-notext.svg.png" width="48"> 

**Python**: The entire code is written in Python.


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/SpaCy_logo.svg/1280px-SpaCy_logo.svg.png" width="48"> 

**Spacy**: An open-source software library for advanced Natural Language Processing (NLP) in Python. It is used for tokenization, named entity recognition (NER), part of speech tagging, and dependency parsing.


<img src="https://avatars.githubusercontent.com/u/388785?s=280&v=4" width="48"> 

**NetworkX**: A Python library used for building and analyzing network graphs.


<img src="https://pbs.twimg.com/profile_images/1366779897423810562/kn7ucNPv_400x400.png" width="48"> 

**Streamlit**: An open-source Python library used to build and run the web application.


<img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Plotly-logo.png" width="80"> 

**Plotly**: This is a graphing library used for creating interactive knowledge graph visualizations.


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1280px-Pandas_logo.svg.png" width="80"> 

**Pandas**: An open-source data analysis and manipulation tool built on top of the Python programming language.


<img src="https://i.pinimg.com/originals/4c/15/a9/4c15a9b76fb269e21b445715a80a78ab.png" width="40"> 

**Time Module**: A Python module that is used here for tracking processing time.


<img src="https://cdn-icons-png.flaticon.com/512/28/28863.png" width="40"> 

**Python-Docx**: A Python library for creating and updating Microsoft Word (.docx) files.


<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/PDF_file_icon.svg/391px-PDF_file_icon.svg.png" width="40"> 

**Pdfplumber**: Used for extracting text from PDF files. It allows detailed access to text, tables, and metadata in PDFs.


<img src="https://upload.wikimedia.org/wikipedia/commons/2/2c/Requests-logo.png" width="48"> 

**Requests**: A simple HTTP library for Python, used to send HTTP requests easily.


<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2021/08/54889soup.png" width="80"> 

**Beautiful Soup (bs4)**: A Python library used here to parse HTML content.

# Contributing

Contributions to enhance <a href= "http://wordlit.net"> Wordlit.net </a> are welcome. Feel free to fork the repository, make changes, and create a pull request.

# License

All code contributed to <a href = "http://wordlit.net">Wordlit.net</a> © 2024 by <a href = "https://www.linkedin.com/in/sahir-maharaj/">Sahir Maharaj</a> is licensed under <a href = "http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1">Attribution 4.0 International<a/>

When using the code from Wordlit.net, please credit as follows: 

`Code sourced from Wordlit.net, authored by Sahir Maharaj, 2024.`

# Contact

**Report a bug or request a feature**: <a href = "mailto:sahir@sahirmaharaj.com">sahir@sahirmaharaj.com</a>

**LinkedIn**: <a href = "https://www.linkedin.com/in/sahir-maharaj/">Sahir Maharaj</a>
