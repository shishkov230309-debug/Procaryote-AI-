# Procaryote-AI-
# DISCLAIMER! The code for the training has been written with heavy AI use
## Reproducible environment

Install the pinned dependencies from `requirements.txt` with Python 3.13.2. The pins were verified against the current Windows environment, including PyTorch 2.7.1 with CUDA 11.8 support.

## Datasets Used 

I used two datasets :
- Du, J., Yang, C., Sun, M. et al. High-Resolution Colony Images of Clinically Isolated Bacteria for Automated Detection and Deep Learning. Sci Data 13, 757 (2026). https://doi.org/10.1038/s41597-026-07095-5. The material from the article hasnot been changed. Creative commons license : https://creativecommons.org/licenses/by-nc-nd/4.0/ 

- Makrai, L., Fodróczy, B., Nagy, S.Á. et al. Annotated dataset for deep-learning-based bacterial colony detection. Sci Data 10, 497 (2023). https://doi.org/10.1038/s41597-023-02404-8. The material has been used without any change. Creative Commons licence :  https://creativecommons.org/licenses/by/4.0/


## Introduction
This project is my initiation to Artificial Intelligence. I initially wanted to fine tune an AI vision model , which would be useful for quick bacteria identification without special chemical test or a microscope. However, only later I understood how naive my idea was, as the form of colonies on agar reflects very little the nature of the bacteria. Moreover, side factors such as colour of agar or the way bacteria was put on it, create a strong need of diversified, rich dataset which I ould not find. The final result is incapable of classifying any of photos that do not belong to the dataset. However, I learned a lot about domestic model fine tuning and I loved the process of fine tuning the model, no matter the result. I will summarise what tools I learned while making this project.

## Things I learned 
First of all, I got a hands-on experience with how the data for AI training is processed. I learned how to split data into three groups : training, testing, and validation sets. I also explored the creation of CSV files for AI training. Even though the majority of my code has been written by AI agents, I encountered major problems that I had to solve, for example I had to centralize the training data from two different datasets, and I had to append indexes of the previous dataset to the new one. 

It was also my first time working with Hugging Face. I downloaded **Google's VIT for image classification** model

What I did is reconfigure the decoder layer of the model to math the number of the bacteria species in the first, bigger dataset. My first training pattern was pretty straightforward, but in latest versions I added some different experiments, as my entire dataset was slightly overfitted for **Staphylococcus Aureus**, so I tried weighted training to resolve this, but without any success. I would follow up on this, but I think I will abandon this project to pursue more interesting and promising ones. 

With, again, help from AI, I added some testing features, such as **thresholding.py** which establishes the minimum confidence level for the model