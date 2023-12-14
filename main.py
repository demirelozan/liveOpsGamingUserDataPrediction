import pandas as pd
from plotting import generate_plots


def main():
    fileLocation = "C:\\Users\\ozand\\Documents\\Python " \
                   "Projects\\liveOpsGamingUserDataPrediction\\ml_project_2023_cltv_train.xlsx"
    data = pd.read_excel(fileLocation, engine='openpyxl')
    generate_plots(data)


if __name__ == "__main__":
    main()
