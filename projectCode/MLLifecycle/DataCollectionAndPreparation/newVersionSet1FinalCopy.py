# Necessary imports
import pandas as pd;
import numpy as np;
import yfinance as yf; #<-- Needed to access Yahoo Finance Dataset(s)
# Below is needed to get the real-time data
from datetime import datetime, timedelta, date
# UPDATE: timedelta will be used to programmatically do the optimality algorithm. Refer to notes on timedelta class for reference.
import pdb as pb
# End of Necessary imports
global seriesVersionOn
seriesVersionOn = True
global resultantDataFrame
resultantDataFrame: pd.DataFrame = pd.DataFrame(columns=["P/B", "P/E", "NCAV", "Date For Eval", "Company", "Optimality"])
resultantDataFrame["P/B"].astype("Float64")
resultantDataFrame["P/E"].astype("Float64")
resultantDataFrame["NCAV"].astype("Float64")
class subsys1:
    def __init__(self,param1 = """Insert any params that may be sufficient here!"""):
        print("---Subsystem 1 in Progress---")
    # component a) [mapping formulas using dep vars and indep vars]
    def compA(self,company, customDateDelta = None):
        PRatioBug = True
        print("--Component a[Subsystem 1] in progress--")
        global arrOfDataFramesNeeded
        arrOfDataFramesNeeded = []
        # Obtaining the dataframes from the yfinance api for company i
        print("-Obtaining the DataFrames from the yfinance api for company i-")
        timePeriod = "4mo"
        timeInterval = "1mo"
        # Obtaining real-time data
        if (company == None):
            company = "MSFT"
        # companyAlias = company or "MSFT"
        companyAlias = company
        ticker = yf.Ticker(f"{companyAlias}")
        global end_date #<-- Setting global so end_date can be accessed.
        customPull = True if customDateDelta != None else False 
        # NOTE: Make sure, customDateDelta is measured in DAYS not YEARS!
        if(customPull == False):
            start_date: datetime = datetime.today() - timedelta(days=10) ; end_date = datetime.today()
        else:
            start_date: datetime = datetime.today() - timedelta(days=customDateDelta) ; end_date = datetime.today()
        #^^ Setting start_date and end_date as arbitrary value relative to current day and current day respectively.
        yearsRelToDateInQ: datetime =  start_date - timedelta(days=5*365)
        start_date = start_date.date().isoformat()
        end_date = end_date.date().isoformat()
        historical_data = ticker.history(period=timePeriod, interval=timeInterval).tz_localize(None) if start_date == None and end_date == None else ticker.history(start=start_date, end=end_date).tz_localize(None)
        print(historical_data)
        # end of Obtaining real-time data
        # Obtaining historical data
        quarterlyBalanceSheet = ticker.quarterly_balance_sheet.T
        arrOfDataFramesNeeded = [historical_data, quarterlyBalanceSheet]
        print(quarterlyBalanceSheet)
        print(quarterlyBalanceSheet.columns)
        conditionForDataFrame2 = ticker.history(period="5y").tz_localize(None) #<-- Here, I set start date to first tuple's date from historical_data var and then get the data 5 yrs BACK relative to that date.
        balanceSheet = ticker.balance_sheet.T
        print(conditionForDataFrame2)
        arrOfDataFramesNeeded.append([conditionForDataFrame2])
        # end of Obtaining historical data
        print("-End of Obtaining the DataFrames from the yfinance api for company i-")
        # CHECKPOINT #1: At this point, datafmres will be obtained to begin process of assigning formulas.
        # End of Obtaining the dataframes from the yfinance api for company i
        boolExpMain = """Can insert a bool exp that checks for number of values in company column, indicating that dataFrame still exists"""
        alreadyExists = False if boolExpMain == True else True
        print("-Beginning of assigning formulas-")
        global independentVars
        independentVars = ["Close" if PRatioBug == False else "Open","Tangible Book Value","High","Retained Earnings","Net Tangible Assets", ]
        global DependentVars
        DependentVars = ["P/B", "P/E", "NCAV", "Total Debt", "Tangible Book Value", "Net Debt", "Stockholder's Equity", "Current Liabilities", "Stockholders Equity", "Date For Eval"] #<-- PN: Date for Eval was created to reference the date to use to make relevant decisions.
        DependentVars.append("Share Price")
        global dataFramesToBeShipped
        dataFramesToBeShipped = [pd.DataFrame(), pd.DataFrame()]
        DependentVars.append("Company")
        # NOTE: alreadyExists prevents currentDataframe from being updated allowing other companies to be added to resultant dataframe.
        # Assigning the formulas
        print("-Assigning formulas-") # PARTITION #1: Need to split code off when porting to notebook here!
        pullingFromTickerInfo = True
        if not PRatioBug:
            dataFramesToBeShipped[0].loc[0,DependentVars[0]] = arrOfDataFramesNeeded[0].loc[arrOfDataFramesNeeded[0].index[0].date().isoformat(),independentVars[0]]/arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]]
            print("---")
            print(arrOfDataFramesNeeded[0].columns)
            print("---")
            print(arrOfDataFramesNeeded[0].loc[arrOfDataFramesNeeded[0].index[0].date().isoformat(),independentVars[0]])
            print(arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]])
            dataFramesToBeShipped[0].loc[0,DependentVars[1]] = arrOfDataFramesNeeded[0].loc[arrOfDataFramesNeeded[0].index[0].date().isoformat(),independentVars[2]]/arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[3]]
            dataFramesToBeShipped[0].loc[0,DependentVars[2]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[4]] #<-- Here, NCAV comes from balance sheet.
            print(dataFramesToBeShipped[0])
            # end of Assigning the formulas
        elif PRatioBug and pullingFromTickerInfo == True:
            # print("---DEBUGGING CHECKPOINT: Seeing what arrOfDataFramesNeeded References---")
            # pb.set_trace()
            dataFramesToBeShipped[0].loc[0,DependentVars[0]] = ticker.info["priceToBook"]
            print("---")
            print(arrOfDataFramesNeeded[0].columns)
            print("---")
            print(ticker.info["regularMarketPrice"])
            print(arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]])
            dataFramesToBeShipped[0].loc[0,DependentVars[1]] = ticker.info["regularMarketPrice"]/ticker.info["earningsQuarterlyGrowth"]
            dataFramesToBeShipped[0].loc[0,DependentVars[2]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[4]] #<-- Here, NCAV comes from balance sheet.
            dataFramesToBeShipped[0].loc[0,DependentVars[3]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),arrOfDataFramesNeeded[1].columns[3]] #<-- Here, NCAV comes from balance sheet.
            dataFramesToBeShipped[0].loc[0,DependentVars[4]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),arrOfDataFramesNeeded[1].columns[4]] #<-- Here, NCAV comes from balance sheet.
            dataFramesToBeShipped[0].loc[0,DependentVars[5]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),arrOfDataFramesNeeded[1].columns[2]] #<-- Here, NCAV comes from balance sheet.
            dataFramesToBeShipped[0].loc[0,DependentVars[6]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),arrOfDataFramesNeeded[1].columns[12]] #<-- Here, NCAV comes from balance sheet.
            dataFramesToBeShipped[0].loc[0,DependentVars[7]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),arrOfDataFramesNeeded[1].columns[29]] #<-- Here, NCAV comes from balance sheet.



            dataFramesToBeShipped[0].loc[0,DependentVars[3]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[4]] 
            dataFramesToBeShipped[0].loc[0,DependentVars[4]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[4]] 


            dataFramesToBeShipped[0].loc[0,DependentVars[4]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[4]] 
            dataFramesToBeShipped[0].loc[0,DependentVars[5]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[4]] 
            print(dataFramesToBeShipped[0])
            dataFramesToBeShipped[0].loc[0,"Share Price"] = ticker.info["regularMarketPrice"]
            # end of Assigning the formulas
        else:
            print("--DEBUGGING CHECKPOINT: Adressing behavior causing error involving Share Price issue---")
            dataFramesToBeShipped[0].loc[0,DependentVars[0]] = ticker.info["regularMarketPrice"]/arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]]
            print("---")
            print(arrOfDataFramesNeeded[0].columns)
            print("---")
            print(ticker.info["regularMarketPrice"])
            print(arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]])
            dataFramesToBeShipped[0].loc[0,DependentVars[1]] = ticker.info["regularMarketPrice"]/arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[3]]
            dataFramesToBeShipped[0].loc[0,DependentVars[2]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[4]] #<-- Here, NCAV comes from balance sheet.
            print(dataFramesToBeShipped[0])
            # end of Assigning the formulas
        print("-End of assigning formulas-")
        if(seriesVersionOn):
            dataFramesToBeShipped[0].loc[0, "Company"] = companyAlias
        print("--Entering Debuging Mode for Checkpoint #2--")
        # CHECKPOINT #2: At this point, formulas will be assigned to neccessary dependent variables for datafrmae that will be output.
        print(dataFramesToBeShipped[0].columns)
        print("--End of Component a[Subsystem 1]--")
        resultantSeries = dataFramesToBeShipped[0]
        # Body of casting dates to cause assignment to be successful
        conditionForDataFrame2.index = conditionForDataFrame2.index.year
        balanceSheet.index = balanceSheet.index.year
        # End of Body of casting dates to cause assignment to be successful
        conditionForDataFrame2["P/E"] = conditionForDataFrame2[independentVars[0]]/balanceSheet.loc[:, "Retained Earnings"]
        conditionForDataFrame2 = conditionForDataFrame2[conditionForDataFrame2["P/E"].notnull()]
        newVersion = True
        return (dataFramesToBeShipped if seriesVersionOn == False else resultantSeries) if newVersion == False else (dataFramesToBeShipped if seriesVersionOn == False else [resultantSeries, conditionForDataFrame2])
        # end of component a)
        # component b) [Populating the new dataframe(s) and preparing dataFrames to be written to files[PN: Will mean that resultant dataframes must be copied to nonlocal variables so subsystem1 can access them(UPDATE: Decided to return the array of these dataframes instead)]
        # end of component b)
    # NOTE: Below was replaced by adding system to class's constructor!

class subsys2:
    def __init__(self,param1 = """Insert any params that may be sufficient here!"""):
        print("---Subsystem 2 in Progress---")
        # self.compA()
    # component a)[creating algorithm for assigning optimality for each company's value]
    def compA(self, param1 = """Insert any params that may be sufficient here!""", arrOfCompanies = []):
        # NOTE: Need to add a condition that ensures that dataframe is updated based on companies passed in.
        # UPDATE: replacing compA with version where only each company is required.
        global dataFrameReffingCompanyData
        version0 = False
        if version0 == True:
            for i in range(iterators):
                print("--Component a[Subsystem 2] in progress--")
                # Body of executing algorithm for strategy responsible for assigning optimality[NOTE: This only applies to retrievedDataFrame[0] since that'll be the training set one]
                # predicate wffs for conditions: i) stock(P/B) <= 1.0, ii) stock(P/E) < max(stockPE(P/E,5)), iii) stock(share price) < 0.67*tangible per-share book value[which can be found in historical_data], iv)  (cont here). where stockPE(x,y) = stock's x ratio in the past 5 years and returns z_i, where z_i = x ratio in year i and i \le y .  [predicate wffs written!]
                # Body of creating boolExp Array for respective conditions
                arrOfCompanies[i] = param1
                currCompany = arrOfCompanies[i]
                dataFrameWithCompanyAsAColumnId = subsys1.compA(company=currCompany)[0] #<-- Returns the dataframe referencing company data.
                dataFrameReffingCompanyData = dataFrameWithCompanyAsAColumnId
                dateInQuestion: datetime = end_date.fromisoformat(end_date)
                yr = 365;
                yearsRelToDateInQ: datetime = dateInQuestion - timedelta(days=5*yr)
                    # Obtaining stocks highest P/E over previous five years
                    # end of Obtaining Stocks highest P/E over previous five years
                arrForCondTwo: pd.DataFrame = arrOfDataFramesNeeded[len(arrOfDataFramesNeeded) - 1] #<-- Did this since I added the dataframe specifically for cond 2 at the END of arrOfDataFramesNeeded. [UPDATE: Need to replace this]
                highestP_EOvrFiveYrs: float = arrForCondTwo["P/E"].max() #<-- REPLACEMENT PENDING: will be replaced when above is filled in. [UPDATE: Using default operator logic to set val to trusy val if ither operand is undef. Will involve querying dataframe for maximum val in column]
                    # Body of creating boolExp Array for respective conditions
                # debtToThingComparison = 
                averageP_EForIndustry = 40.35 #<-- average P/E ratio for tech comps.
                print("--DEBUGGING CHECKPOINT #1: Evaluating Bool expressions ensuring the classifications are doing what they are supposed to---")
                boolExps: list[bool] = [
                        dataFrameReffingCompanyData["P/B" if currCompany == None else ("P/B",currCompany)] <= averageP_EForIndustry,
                    dataFrameReffingCompanyData["P/E" if currCompany == None else ("P/E",currCompany)] <= 0.4*highestP_EOvrFiveYrs,
                    dataFrameReffingCompanyData["Total Debt"] > dataFrameReffingCompanyData["Tangible Book Value"],  
                    dataFrameReffingCompanyData["Total Debt"] > 2*dataFrameReffingCompanyData["NCAV"],  
                    dataFrameReffingCompanyData["Total Debt"] > 2*dataFrameReffingCompanyData["Current Liabilities"] and dataFrameReffingCompanyData["Total Debt"] > 2*dataFrameReffingCompanyData["Stockholders Equity"],  
                    
                ]
                    # End of Body of creating boolExp Array for respective conditions
                 
                # debtToThingComparison = 
                # debtToThingComparison =
                score = 0
                if boolExps[0]:
                    score += 1
                if boolExps[1]:
                    score += 1
                if boolExps[2]:
                    score += 1
                if boolExps[3]:
                    score += 1
                if boolExps[4]:
                    score += 1
                dataFrameReffingCompanyData["Optimality"] = score;
                # End of Body of executing algorithm for strategy responsible for assigning optimality
        else:
            # Version of code where only company[i] is proprioritezed and a series is returned with optimality there.
            print("--Component a[Subsystem 2] in progress--[VERSION #2]")
            currCompany = param1
            arrOfDataFramesNeeded = subsys1().compA(company=currCompany)
            dataFrameWithCompanyAsAColumnId = arrOfDataFramesNeeded[0]
            dataFrameReffingCompanyData = arrOfDataFramesNeeded[1]
            arrForCondTwo = arrOfDataFramesNeeded[1]
            # Body of version where compA(subsys1) returns TWO things
            dateInQuestion: datetime = datetime.fromisoformat(end_date)
            yr = 365;
            yearsRelToDateInQ: datetime = dateInQuestion - timedelta(days=5*yr)
                # Obtaining stocks highest P/E over previous five years[for assistance, use this search query: `? does ticker object have a start date parameter`]
                # end of Obtaining Stocks highest P/E over previous five years
            highestP_EOvrFiveYrs = arrForCondTwo["P/E"].max()
            dataFrameThatRefsSharePrice = dataFrameWithCompanyAsAColumnId #<-- This variable is self explanatory. Need to modify soon.
            """ boolExps: list[bool] = [
                    dataFrameReffingCompanyData["P/B"] <= 1.0,
                dataFrameReffingCompanyData["P/E"] <= 0.4*highestP_EOvrFiveYrs,
                dataFrameReffingCompanyData["Share Price"] < 0.67*dataFrameThatRefsSharePrice["Tangible Book Value"]
            ] """
            """ boolExps: list[bool] = [
                    dataFrameWithCompanyAsAColumnId["P/B"] <= 1.0,
                dataFrameWithCompanyAsAColumnId["P/E"] <= 0.4*highestP_EOvrFiveYrs,
                dataFrameWithCompanyAsAColumnId["Share Price"] < 0.67*dataFrameThatRefsSharePrice["Tangible Book Value"]
            ] """
            boolExps: list[bool] = [
                    dataFrameWithCompanyAsAColumnId["P/B"] <= 1.0,
                dataFrameWithCompanyAsAColumnId["P/E"] <= 0.4*highestP_EOvrFiveYrs,
                dataFrameWithCompanyAsAColumnId["Share Price"] < 0.67*(dataFrameWithCompanyAsAColumnId["Share Price"]/dataFrameThatRefsSharePrice["P/B"])
            ]
            # UPDATE: Problem originates from dataFrameThatRefsSharePrice["Tangible Book Value"][UPDATE: Problem fixed!]
                # End of Body of creating boolExp Array for respective conditions
            score = 0
            if boolExps[0].any():
                score += 1
            if boolExps[1].any():
                score += 1
            if boolExps[2].any():
                score += 1
            dataFrameWithCompanyAsAColumnId["Optimality"] = score; # UPDATE: Works as intended. Focu sshould shift back towards ensuring that Share Price thing works properly.
            # End of Body of executing algorithm for strategy responsible for assigning optimality
            print("--End of Component a[Subsystem 2] in progress--")
            return dataFrameWithCompanyAsAColumnId
# end of component a)
    # component b)
    def compB():
        print("--Component b[Subsystem 2] in progress--")
        global retreivedDataFrames
        retreivedDataFrames = [dataFrameReffingCompanyData,dataFrameReffingCompanyData] or pd.DataFrame()
        # Body of writing resp dataframes to files[need to use to_csv I believe]
        filePathToTrainingSetDir: str = "MLLifecycle/ModelDevelopmentAndTraining/TrainingSets"
        filePathToTestSetDir: str = "MLLifecycle/ModelDevelopmentAndTraining/TestingSets"
        print("--End of Component b[Subsystem 2]--")
        print("Entering Debuging Mode for Checkpoint #3")
        # CHECKPOINT #3: At this point, dataframes will be written to the neccessary files to be ingested by the Model.
        print("--Writing Dataframes for Model Ingestion--")
        retreivedDataFrames[0].to_csv(f'{filePathToTrainingSetDir}/trainingSet{setNum if setNum != None else 1}')
        retreivedDataFrames[1].to_csv(f'{filePathToTrainingSetDir}/testSet{setNum if setNum != None else 1}')
        # End of Body of writing resp dataframes to files
        print("--End of Writing Dataframes for Model Ingestion--")
        print("--End of Component b[Subsystem 2]--")
        return #<-- Testing version of return statement[anything above this return statement is sucessful and everything below hasn't been tested yet. THis is relative to each function]
    # end of component b)

def main(companies: list[str] = [], desiredDateToPullInvestment = None):
    print("---Starting Data Prep Process---")
    # print("---DEBUGGING CHECKPOINT #1: Investigating companySeries value---")
    # pb.set_trace()
    if len(companies) == 0:
        companies: list[str] = ["GOOG","AAPL", "AMZN", "MSFT"] 
    
    listOfSeriesToCreateDataFrame = []
    for i in range(len(companies)):
        print(f"----Adding company {companies[i]} to engineered dataset----")
        # """
        # NOTE: Will uncomment, once everything with the functions used here is situated[add a checklist here: ]
        print("---Starting Subsystem 1---")
        # Call function referencing topmost subsystem #1 here:
        a: str = ""; b: str = ""
        if desiredDateToPullInvestment == None:
            subsys1().compA(company=companies[i])
            subsys2().compA(param1=companies[i])
        else:
            subsys1().compA(company=companies[i],customDateDelta=desiredDateToPullInvestment)
            subsys2().compA(param1=companies[i])
        retSeries = True
        if retSeries:
            # ^^ NOTE: Above is returning a series each time[at least this is the assumption]
            companySeries = subsys2().compA(param1=companies[i])
            listOfSeriesToCreateDataFrame.append(companySeries)
        else:
            # ^^ NOTE: Above is NOT retruning a series
            print("---Assumption that series is NOT returned---")
        # Will replace above with this: subsys1(company_i)
        print("---End of Subsystem 1---")
        print(f"----End of Adding company {companies[i]} to engineered dataset----")
    
    resultantDataFrame = pd.concat([pd.DataFrame(x) for x in listOfSeriesToCreateDataFrame]).reset_index() #<-- used list comprehension to transform listOfSeries to resultantDataFrame.
    del resultantDataFrame['index']
    print(resultantDataFrame) #<-- THis dataframe will reference the dataframe that adheres to the follwowing object:
    # company(CompanyName, "P/B", "P/E", "NCAV", "Date For Eval", "Optimality", (cont here if applicable))[NOTE: Will be wise to make a Entity via ERDs for documentation when writing paper at end]
    inNoteBook = False
    DemoMode = True
    filePathToModelDir = "C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/projectCode/MLLifecycle/ModelDevelopmentAndTraining/preparedDataset.csv" if inNoteBook == False or DemoMode == True else "preparedDataset.csv"
    # Body of handling edge case where all of them are same optimality
    resultantDataFrame.to_csv(f"{filePathToModelDir}")
    if((resultantDataFrame["Optimality"] == 0).all() == True):
        # Setting optimality column to be based on alphabetical ordering
        resultantDataFrame.sort_values(by='Company',inplace=True)
        resultantDataFrame = resultantDataFrame.set_index(np.arange(4))
        resultantDataFrame.loc[:,"Optimality"] = pd.Series(np.arange(resultantDataFrame["Optimality"].shape[0]))
        # End of Setting optimality column to be based on alphabetical ordering
        resultantDataFrame.to_csv(f"{filePathToModelDir}")
    # End of Body of handling edge case where all of them are same optimality
def dataPrepDeriv(companies: list[str] = [], desiredDateToPullInvestment = None):
    print("---Starting Data Prep Process---")
    # print("---DEBUGGING CHECKPOINT #1: Investigating companySeries value---")
    # pb.set_trace()
    if len(companies) == 0:
        companies: list[str] = ["GOOG","AAPL", "AMZN", "MSFT"] 
    
    listOfSeriesToCreateDataFrame = []
    for i in range(len(companies)):
        print(f"----Adding company {companies[i]} to engineered dataset----")
        # """
        # NOTE: Will uncomment, once everything with the functions used here is situated[add a checklist here: ]
        print("---Starting Subsystem 1---")
        # Call function referencing topmost subsystem #1 here:
        a: str = ""; b: str = ""
        if desiredDateToPullInvestment == None:
            subsys1().compA(company=companies[i])
            subsys2().compA(param1=companies[i])
        else:
            subsys1().compA(company=companies[i],customDateDelta=desiredDateToPullInvestment)
            subsys2().compA(param1=companies[i])
        retSeries = True
        if retSeries:
            # ^^ NOTE: Above is returning a series each time[at least this is the assumption]
            companySeries = subsys2().compA(param1=companies[i])
            listOfSeriesToCreateDataFrame.append(companySeries)
        else:
            # ^^ NOTE: Above is NOT retruning a series
            print("---Assumption that series is NOT returned---")
        # Will replace above with this: subsys1(company_i)
        print("---End of Subsystem 1---")
        print(f"----End of Adding company {companies[i]} to engineered dataset----")
    
    resultantDataFrame = pd.concat([pd.DataFrame(x) for x in listOfSeriesToCreateDataFrame]).reset_index() #<-- used list comprehension to transform listOfSeries to resultantDataFrame.
    del resultantDataFrame['index']
    print(resultantDataFrame) #<-- THis dataframe will reference the dataframe that adheres to the follwowing object:
    # company(CompanyName, "P/B", "P/E", "NCAV", "Date For Eval", "Optimality", (cont here if applicable))[NOTE: Will be wise to make a Entity via ERDs for documentation when writing paper at end]
    inNoteBook = False
    DemoMode = True
    filePathToModelDir = "C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/projectCode/MLLifecycle/ModelDevelopmentAndTraining/preparedDataset.csv" if inNoteBook == False or DemoMode == True else "preparedDataset.csv"
    # Body of handling edge case where all of them are same optimality
    resultantDataFrame.to_csv(f"{filePathToModelDir}")
    if((resultantDataFrame["Optimality"] == 0).all() == True):
        # Setting optimality column to be based on alphabetical ordering
        resultantDataFrame.sort_values(by='Company',inplace=True)
        resultantDataFrame = resultantDataFrame.set_index(np.arange(4))
        resultantDataFrame.loc[:,"Optimality"] = pd.Series(np.arange(resultantDataFrame["Optimality"].shape[0]))
        # End of Setting optimality column to be based on alphabetical ordering
        resultantDataFrame.to_csv(f"{filePathToModelDir}")
    # End of Body of handling edge case where all of them are same optimality
dataPrepDeriv(['GOOG', 'AAPL', 'AMZN', 'MSFT', 'META', 'OTIS', 'HOLX', 'CHRW', 'FITB', 'PANW'])