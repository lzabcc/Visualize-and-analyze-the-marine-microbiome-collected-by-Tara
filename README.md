Project Overview

This project is aimed at designing and implementing an interactive multi-coordinated view visualization tool to analyze ocean microbiome data. The tool is intended to assist microbiologists and other biological science researchers in gaining insights into marine microbial communities, specifically focusing on exploring the abundance patterns and diversity of microorganisms in different ocean regions.

Data Description

The dataset comprises 135 samples with 35,650 operational taxonomic units (OTUs), representing the detection levels of microbes. These OTUs are classified hierarchically according to taxonomy, with each OTU associated with a unique identifier and relative abundance percentages for each sample. Additional metadata include sampling year, month, latitude, longitude, depth, and regional classifications.

Key Features

Interactive Visualization: The tool supports interactive exploration of microbial abundance and diversity patterns.

Multiple Coordinated Views: Incorporates multiple views synchronized to facilitate comparative analysis.

Contextual Understanding: Integrates sample metadata to contextualize abundance patterns.

Overview and Detail: Allows users to gain an overview while also enabling detailed investigation of specific patterns.

Implementation Details

The implementation involved several key steps:

Data Preparation: Preprocessing and cleaning the provided OTU tables and metadata.

Visualization Design: Crafting a multi-coordinated view system using Python with Altair and Pandas libraries.

Interactivity: Adding interactive elements to enhance user engagement and data exploration.

Accessibility: Ensuring the visualization is accessible and adaptable to diverse users.

Technologies Used

Python: Primary programming language for data processing and visualization.

Altair: Library for generating interactive visualizations.

Pandas: Library for data manipulation and analysis.

HTML: Final visualization output format for web compatibility.

User Interface

The user interface consists of multiple coordinated views, each serving a specific purpose:

Abundance Heatmap: Displays the relative abundance of OTUs across samples.

Geographical Map: Integrates sample locations with abundance patterns.

Histograms and Scatter Plots: Provides detailed distribution and correlation analysis.
