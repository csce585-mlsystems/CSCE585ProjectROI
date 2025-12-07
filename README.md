# Project Title  
## Generating Passive Income For Young Adults Through a Value Investing Machine Learning Model  

## Group Info  
- Ardoine Docteur
  - Email: adocteur@email.sc.edu   
- Ava Patel
  - Email: avamp@email.sc.edu  
- Michelle Ihetu
  - Email: mihetu@email.sc.edu  

## Project Summary/Abstract  
This project is a machine learning model that helps young adults generate passive income through the strategy of value investing.
By combining financial data from sources such as Yahoo Finance, Alpha Vantage, and The Federal Reserve Economic Database (FRED), the model will analyze value metrics such as the Price to Earnings (P/E) ratio, the Price to Book (P/B) ratio, the Debt to Equity (D/E) ratio, and the Free Cash Flow (FCF) yield. The system will take in user input such as investment amount, desired return amount, and a time frame to generate stock reccomendations. It will then compare those recommendations to benchmarks like the S&P 500 and Russell 1000 Value Index to measure performance.

## Problem Description  
Many young adults want to start investing but don’t really know where to start or simply don't feel confident enough in their decisions to try. The stock market can seem confusing and risky, especially to beginners. Our project makes it that much easier for new investors by using a machine learning model to give users simple, data driven, and safer investment suggestions.
- Motivation  
  - Young adults want to accumulate a passive income overtime but do not know where to start. 
  - People do not have time to research in depth and learn what everything is when it comes to investing.  
  - Using ML will aid the user in making a decision in what to invest in by providing a suggestion backed with easy to understand data. This will help the user understand what exactly they are looking at.  
- Challenges  
  - Collecting and cleaning accurate and up to date financial data from many different sources.  
  - Making sure that the model is properly suggesting the undervalued stocks.
  - Coming up with a clear way for users to input their informattion and understand their data.

## Contribution  
 

### ['Replication Of Existing Work'],[`Extension of existing work`]  
We build upon Machine Learning Financial Analysis [Reference 3] to reproduce their work regarding the application of financial ratios and market indicators to predict undervalued stocks and generate above average returns. Specifically, we replicate their method of using metrics such as P/E, P/B, D/E, FCF to evaluate company value and forecast future performance.

We extend this work in the following ways: 

- Integrating APIs to automatically collect and clean real time financial and economic data
- Combining traditional value metrics with machine learning models to identify nonlinear patterns and improve prediction accuracy
- Comparing model performance against market benchmarks such as the S&P 500 and Russell 1000 Value Index to test investment effectiveness
- Designing a beginner friendly interface that provides clear investment recommendations and visualizations for beginners
- Including educational explanations that define financial terms and guide users through understanding why certain stocks are chosen over others

Note: This has no impact on your grade as long as you properly follow the procedure (e.g., problem identification, motivation, method, discussion, results, conclusion). Choosing [`Novel contribution`] will not give you any advantage over [`Replication of existing work`].  

## References  
### BibTeX entries for all sources used in this project are available in the [references.bib](docs/references.bib) file.

[1] Yan, K., & Li, Y. (2024). Machine learning-based analysis of Volatility Quantitative Investment Strategies for American Financial Stocks. Quantitative Finance and Economics, 8(2), 364–386. https://doi.org/10.3934/qfe.2024014

[2]Kirkpatrick, T., & LaGrange, C. (n.d.). Robotic Surgery: Risks vs. Rewards. PSNet. https://psnet.ahrq.gov/web-mm/robotic-surgery-risks-vs-rewards#:~:text=RAS%20shares%20the%20same%20risks,pelvis%20adequately%20to%20perform%20RALP. 

[3] Cao, K., & You, H. (2022, September 28). Fundamental Analysis Via Machine Learning: Financial Analysts Journal: Vol 80, no 2. Taylor & Francis Online. https://www.tandfonline.com/doi/abs/10.1080/0015198X.2024.2313692

## Dependencies  
### Include all dependencies required to run the project.  
- Python 3.11+
- macOS 13+ or Windows 10+
- uv
- uv.lock and pyproject.toml are included in this repository
  
For Python users: Please use [uv](https://docs.astral.sh/uv/) as your package manager instead of `pip`. Your repo must include both the `uv.lock` and `pyproject.toml` files.  

## Directory Structure  
```
├── README.md
├── UIDesignDescs
│   └── set1.md
├── UXResearch
│   └── purpose.md
├── __pycache__
│   └── constants.cpython-311.pyc
├── app.py
├── constants.py
├── docs
│   ├── p0
│   │   ├── P0.pdf
│   │   └── SlidesP0.pdf
│   ├── p1
│   │   ├── SlidesP1.pdf
│   │   └── p1.txt
│   ├── p2
│   │   └── p2.txt
│   └── references.bib
├── milestones
│   ├── milestone1
│   │   ├── Experiments
│   │   ├── p1Document.md
│   │   └── powerpointTemplate.tex
│   └── milestone2
│       ├── guide.md
│       ├── initialExperiments.ipynb
│       └── initialExperimentsFinalCopyP2Version.ipynb
├── preferences.json
├── projectCode
│   ├── MLLifecycle
│   │   ├── DataCollection
│   │   ├── DataPreparation
│   │   ├── ModelDevelopment
│   │   ├── ModelTraining
│   │   └── summaryOfFiles.md
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   ├── components
│   │   ├── State0
│   │   ├── State1
│   │   ├── State2
│   │   ├── __init__.py
│   │   └── __pycache__
│   ├── requirements.md
│   └── routes
│       ├── State0Routes.py
│       ├── State1Routes.py
│       ├── State2Routes.py
│       ├── __init__.py
│       └── __pycache__
├── pyproject.toml
├── results.py
├── run.py
├── users.json
└── uv.lock
```

⚠️ Notes:  
- All projects must include the `run.<ext>` script (extension depends on your programming language) at the project root directory. This is the script users will run to execute your project.  
- If your project computes/compares metrics such as accuracy, latency, or energy, you must include the `result.<ext>` script to plot the results.  
- Result files such as `.csv`, `.jpg`, or raw data must be saved in the `data` directory.  

## How to Run  
- Include all instructions (`commands`, `scripts`, etc.) needed to run your code.  
- Provide all other details a computer science student would need to reproduce your results.
- Some exploratory plots in the notebooks use matplotlib, which may require additional system level setup on macOS with Python 3.13. The core application and ML pipeline run without it.

Example:  
- Install uv & activate environment (MAC)
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv sync
  source .venv/bin/activate
  ```

- Install uv & activate environment (WIndows)
  ```bash
  iwr https://astral.sh/uv/install.ps1 -useb | iex
  uv sync
  .\.venv\Scripts\activate
  ```

- To run the app:  
  ```bash
  python run.py
  ```

- The results:  
  ```bash
  python results.py
  ```
  
- To reproduce ML experiment notebooks:  
  ```bash
  cd milestones/milestone1 
  cd milestones/milestone2

  jupyter notebook
  ```
  
- To run the files directly:  
  ```bash
  python projectCode/MLLifecycle/ModelDevelopment/baseline_model.py
  python projectCode/MLLifecycle/ModelDevelopment/experiment1_model.py
  ```

## Demo  
- [Link to Video](https://youtu.be/9m3HA3epPpU)
- In the video, we are doing the following steps: 1) Logging into the application with credentials, 2) Setting desired amount to invest and setting the optimal amount of money to result from said investment, 3) We press 'Generate Recommendations', which accesses a repository of stocks and ranks these stocks based on the value investing strategy, and 4) A screen appears referencing the results of the ML Model's predictions. 
---
