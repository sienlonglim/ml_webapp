# Machine Learning Web App :house:
This project utilises open data from <a href="https://data.gov.sg/">Data.gov.sg</a> to build several Machine Learning (ML) models that help predict HDB Resale Prices. <a href="https://natuyuki.pythonanywhere.com/">Live project</a>

The main focus of this project is to complete a full cycle of
- Extract Transform Load (ETL)
- ML Model building
- Deployment
- Live dashboarding

The project involves a large dataset (>40k points) involving geodata of all Singapore HDB resale prices over the years 2022 and 2023. 

The following steps were taken in the project: (all steps can be found in the JupyterNotebook ipynb files)

1. Data was obtained through rest API calls to Data.gov.sg, followed by data wrangling
2. Feature creation and selection (using KBest on Mutual Information, L1 Regularisation)
3. Hyperparameter tuning (Random Cross Validation)
4. Model selection and testing Normal and Ensemble models (Gradient boosting, Random forest)
5. Front end web application (Flask) development with Bootstrap 5
6. Dashboarding (Tableau & Streamlit)

![snippet](https://github.com/sienlonglim/ml_webapp/assets/109359023/c7eea635-3944-4bc8-a361-eceea3492647)


