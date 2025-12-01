# NOTE: This file will be built based on the contents of the purpose.md file. As of now, the amount of functiosn here, should be the number of subsystems, and then those subsystems' steps should have their own functions. This will help with doing component-based debugging opposed to dealing with all dependencies at once. If functions are nested, make sure to denote them as such. 

# Reports: [10/23/25] part 1) At this point, finished writing abstract rep of how prog will acheive goal. Started using a functional prog format to acheive goals. In next sesh, focus should be on finishing up setting up coding env programatically FIRST, and then inserting the neccessary pandas stuff, whilst simulatnatneously doing more research on pandas as well. part 2) Continued learning pandas stuff, and continued seting up code env programatically. In next sesh, continue the aforementioned process. When significant prog is made, begin inserting code related to data engineering. Also, when establishing pipeline this search query may be useful: `start chrome "? is the relative path for a filepath in python script relative to the place it is executed?"`. [10/24/25] part 1) Continued seting of programmatic env. At this point, I believe Subssytem2's programmatic rep should be worked on next, whilst simulataneously implementing more pandas-specific code. part 2) Continued working on programming code. Adhere to aforemetioend request for next session. part 3) Starting setting up Subsystem2's programmatic representation. In next sesh, make sure to implement a quick algo that seraches for the right table to pull from both the real-time data and the historical data, in addition to the aforementioned tasks from previous report(s). part 4) Started process for integrating external dataframes. Did not adhere to aformentioned objs directly. In next sesh, make sure to adhere to aformentioned objectives. part 5) Started wriritng logic code to facilitate optimality algo. Also worked on algo for obtaining the right attrs from the Ticker Object. In next sesh, take time to finish up Pandas tutorial. part 6) Continued where I left off. In next sesh, start testing out certain portions of program. part 7) Got the algorithms implemented, began testingg parts of program. Now, the only problem is having the varaibels reference the right columns. These places are marked with REPLACEMENT PENDING. In next sesh, go back to learning some more pandas and then work on finishing up the rest of the stuff. May be wise to jump to Time Series data chapter from Python for Data Analysis Textbook to interface with dates properly.
# Reports: [10/26/25] part 1) Made more progress by fixing contents relative to error. Had to put the dataframes needd into the python list to be accessed later in the script. In next sesh, continue this process and addresss the 'REPLACEMENT PENDING' notes and make neccessary replacements. part 2) Added a few more things. Added nomenclature for documentation w.r.t comments, to ease the process of problem solving. Addressed a few of the requirements. part 2) Started testing other stuff. Didn't do anything related to time series learning for pandas. In next sesh, focus on learning pandas from geeksforgeeks and going through times series chapter from that textbook. Then, finish this file to integrate to model with this egnienered dataset. Also, there is an error, use code output for reference. part 3) Found some logistical issues wiht some parts. Started process of getting 2nd cond implemented. Really, want to get this proj finished by end of 10/27/25 to get a decent grade.  [10/27/25] part 1) Worked on outermost pipeline script. Going forward, working in reverse order from end of data prep pipeline to beginning to ensure that the nested function(s) and function(s) work properly. part 2) Have some issues getting condition 2. part 3) Got dataframe for condition 2. Think now I need to designate the date. Will be KEY for making decisions. part 3) Made a decent amount of progress. Used comments to update some steps that need to be taken in order to ensure smooth integartion between data prep and model development. In next sesh, continue solving problems to get data prepped for model dev. Particularly, write code that applys the subsystems iteratively to each company. part 4) got subsystem1(compA()) complete, in terms of logic. However, will be imperative to handle presence of null values and omitting them when possible. **PN: I believe it has something to do with the fact that the indices for the row tuples is date-based**. In next sesh, continue to the subsystem2 to ensure that they work logically. part 5) Started working on logic for subsys2. Noticed that I need to handle the creation of dataframe with all companies on it properly. Need to figure out how to do that in subsys2. [10/29/25] part 1) Got the class for subsy2 indented properly for testing. In next sesh, continue runnign tests to ensure that subsy2's components work properly.  [10/31/25] part 1) Started working more on model file to facilitate process for running experiments via Model pipeline. In Next sesh, continue learning more about pandas, and datetime object so you can sync up the times properly to get values lined up to get perimissible data output. Also, may want to make time working on reasons why these milestone were so late. [11/7/25] part 1) Made significant progress. Finally understanding debugging process better now. **Focus is now shifting more towards experimental setups, and creating environment to load images to web application**. 

# Notable things: a) This search query specifies that it's possible to access attributes for a class. THis allows me to programatically identify the stuff that I need opposed to doing things brute force: `start chrome "? is there a list of attribues for a particular class in Python?" `b) Make sure to make copy of this python file and then run it in the MLLifecycle directory. c) NOTE: Think it'd be wise to insert some breakpoints/places to use pdb to ensure that things run as intended. I believe it will make debugging more efficient. d) As of 11/3/25, my new problem is syncing the dates for the data. Iniiat soln consisted of using start and end parameters to specify the bounds needed and then syncing the dates that way. Before this is implemented, I need to ensure that the rest of pipeline is provably correct still so my it soln can be plugged and played. e) As of 11/6/25, the solution is nearly complte. At this point, the problem is ensuring that n companies are allowed to be entered, and ensuring that at the end, each company has its own row-tuple.  f) Everything works up until after checkpoint #2 from a provabiilty perspective as of 11/6/25 . ALSO, NEED TO ADD A STATE THAT ALLOWS USER TO SPECIFY THE STOCKS THEY WANT TO CHOOSE. 

# IMPORTANT: As of 11/22/25: The main focus now is ensuring that the Price from P/B and P/E are pulled from correct data. After that, everything should work properly! To verify, review code again and add any updates here: a) Learned that open should be replaced with closing since, according to net, it refers to the actual price. 
# Necessary imports
import pandas as pd;
import numpy as np;
import yfinance as yf; #<-- Needed to access Yahoo Finance Dataset(s)
# Below is needed to get the real-time data
from datetime import datetime, timedelta, date
# UPDATE: timedelta will be used to programmatically do the optimality algorithm. Refer to notes on timedelta class for reference.  
import pdb as pb #<-- NOTE: Need to use this to test certain sections of code to debug and make neccessary
# End of Necessary imports

# NOTE: Make sure certain vars declared are nonlocal so they can be used in other functions!
global seriesVersionOn
seriesVersionOn = True
global resultantDataFrame
resultantDataFrame: pd.DataFrame = pd.DataFrame(columns=["P/B", "P/E", "NCAV", "Date For Eval", "Company", "Optimality"])
resultantDataFrame["P/B"].astype("Float64")
resultantDataFrame["P/E"].astype("Float64")
resultantDataFrame["NCAV"].astype("Float64")
# resultantDataFrame["Date for Eval"].astype(datetime)

class subsys1:
    def __init__(self,param1 = """Insert any params that may be sufficient here!"""):
        print("---Subsystem 1 in Progress---")
        # self.compA(param1)
    # component a) [mapping formulas using dep vars and indep vars]
    def compA(self,company): #<-- NOTE: Thought about passing in dataframe as a parameter as well for instances where multiple companies need to be added to resultant dataframe. 
        # PRatioBug = False
        PRatioBug = True
        print("--Component a[Subsystem 1] in progress--")
        global arrOfDataFramesNeeded 
        #arrOfDataFramesNeeded: list[pd.DataFrame] = []
        arrOfDataFramesNeeded = []
         
        # Obtaining the dataframes from the yfinance api for company i
        print("-Obtaining the DataFrames from the yfinance api for company i-")
        timePeriod = "4mo"
        timeInterval = "1mo"
        # PN: Since Price, Books will be used will need real-time data. 
        # Obtaining real-time data
        if (company == None):
            company = "MSFT"
        
        companyAlias = company or "MSFT"#<-- For Microsoft[using default operator here to prep function to be generalized to add other companies]
        # pb.set_trace()
        ticker = yf.Ticker(f"{companyAlias}") #<-- NOTE: Will need to replace this with multiple tickers later on. 
        #ticker.info
        #ticker.info.keys() #<-- Cotnains p/b ratio but doesn't have p/e ratio. So, will not be using it. 
        """
        listOfAttrs = dir(ticker)
        for attrib in listOfAttrs:
            # psuedosteps: a) Accessing curr atrib, b) if it rets a dataframe, then checking columns to see if it matches what I Pneed. 
            #for j in range(6): #<-- replacing later on, just a placeholder rn.
           if "High" in getattr(ticker,attrib).columns:
           # Means that the attrib should be used to obtain data needed:
           arrOfAttrsFromExtDataFramesNeeded.append(attrib)
           # ^^ NOTE: Above needs to be generalized[may need a nested for loop before this if statement to go through ALL requested col ids]
        """

        
        global end_date #<-- Set gloabl so end_date can be accessed. 
        start_date: datetime = datetime.today() - timedelta(days=10) ; end_date = datetime.today() #+ timedelta(days=1*365) <-- UPDATE: Took this out b/c start_date needs to begin in past, and end date needs to be in present. 
        yearsRelToDateInQ: datetime =  start_date - timedelta(days=5*365)
        start_date = start_date.date().isoformat()
        end_date = end_date.date().isoformat()
        # UPDATE: Below, I setup environmetn such that, when start_date and end_date are entered, then the pipeline works as expected. 
        # historical_data = ticker.history(period=timePeriod, interval=timeInterval).tz_localize(None) if start_date == None and end_date == None else ticker.history(start=start_date.date().isoformat(), end=end_date.date().isoformat()).tz_localize(None) <-- commented out for obvious reasons. 
        historical_data = ticker.history(period=timePeriod, interval=timeInterval).tz_localize(None) if start_date == None and end_date == None else ticker.history(start=start_date, end=end_date).tz_localize(None)
        print(historical_data)
        # end of Obtaining real-time data
        
        # PN: I believe NCAV == Net Tangible Assets
        # Obtaining historical data
        # quarterlyBalanceSheet = ticker.balance_sheet.T 
        quarterlyBalanceSheet = ticker.quarterly_balance_sheet.T 
        # UPDATE[NOTE]: Will need body that queries for row tuples in the start_date and end_date range
        
        arrOfDataFramesNeeded = [historical_data, quarterlyBalanceSheet]
        print(quarterlyBalanceSheet)
        print(quarterlyBalanceSheet.columns)
        # Below involves getting 5 yr data to support condition #2. 
        # ISSUE #1[complete]
        #conditionForDataFrame2 = ticker.history(start=arrOfDataFramesNeeded[0].index[0],period="1y", interval="5y").tz_localize(None) #<-- Here, I set start date to first tuple's date from historical_data var and then get the data 5 yrs BACK relative to that date. 
        conditionForDataFrame2 = ticker.history(period="5y").tz_localize(None) #<-- Here, I set start date to first tuple's date from historical_data var and then get the data 5 yrs BACK relative to that date. 
        balanceSheet = ticker.balance_sheet.T 
        # Body of assigning P/E column

        
        # End of Body of assigning P/E column
        # Below, the dates within the bounds of start date and 5 years succeeding start date index is queried to obtain relevant data to be used later. 
        print("---Setting checkpoint to see what happens to conditionForDataFrame2 variable---")
        # pb.set_trace()
        # conditionForDataFrame2 = conditionForDataFrame2 if start_date == None and end_date == None else conditionForDataFrame2.loc[start_date:yearsRelToDateInQ, :];
        # NOTE: ^^ Above caused an error due to else body of ternary operator. Thus, statement has been commented out. 
        
        # End of body of testing out bounds interval ideas
        print(conditionForDataFrame2)
        # END OF ISSUE #1
        arrOfDataFramesNeeded.append([conditionForDataFrame2])
        # end of Obtaining historical data
        print("-End of Obtaining the DataFrames from the yfinance api for company i-")
        print("Entering Debuging Mode for Checkpoint #1")
        # pb.set_trace() #<-- Will use this   tcheck for vals of variable(s), as well as set certain vars to certain values to cause different behavior.   
        # CHECKPOINT #1: In proof, at this point, datafmres will be obtained to begin process of assigning formulas.           # For Everything else, I just need historical data. Correct me if wrong here: .
        # End of Obtaining the dataframes from the yfinance api for company i

        # Debugging Report [11/13/25] part 1) When running through debugger, found that the columns are assigned properly, BUT some of the computations for the data engineering results in NaN values. Need to figure out why, think it can be solved by syncing datetime(s). Continue progress on line 111 via the debugger. Jump to that line by finding command to jump to line 111. 
        #return
        boolExpMain = """Can insert a bool exp that checks for number of values in company column, indicating that dataFrame still exists"""
        alreadyExists = False if boolExpMain == True else True  #<-- setting to 0 by default
        print("-Beginning of assigning formulas-")
        global independentVars
        
        # independentVars = ["High" if PRatioBug == False else "Share Issued","Tangible Book Value","High","Retained Earnings","Net Tangible Assets"]
        independentVars = ["Close" if PRatioBug == False else "Open","Tangible Book Value","High","Retained Earnings","Net Tangible Assets"]
        global DependentVars
        DependentVars = ["P/B", "P/E", "NCAV", "Date For Eval"] #<-- PN: Date for Eval was created to reference the date to use to make relevant decisions.  
        DependentVars.append("Share Price")
        global dataFramesToBeShipped
        dataFramesToBeShipped = [pd.DataFrame(), pd.DataFrame()]
        DependentVars.append("Company")
        # NOTE: alreadyExists prevents currentDataframe from being updated allowing other companies to be added to resultant dataframe. 
        dataFramesToBeShipped[0] = pd.DataFrame(columns=DependentVars) if alreadyExists == False else dataFramesToBeShipped[0] #<-- TrainingData
        print(dataFramesToBeShipped[0])
        DependentVars.append("Optimality")
        dataFramesToBeShipped[1] = pd.DataFrame(columns=DependentVars) if alreadyExists == False else dataFramesToBeShipped[1] #<-- TestingData
        print(dataFramesToBeShipped[1])
        # Assigning the formulas
        print("-Assigning formulas-")
        # pb.set_trace() <-- COMPLETE![Assinging formulas worked as intended now!]
        # NOTE: When assigning, it makes sense to only apply formula to most recent data from quarterlyBalanceSheet aka the first tuple. 
        # dataFramesToBeShipped[0][DependentVars[0]] = arrOfDataFramesNeeded[0][independentVars[0]]/arrOfDataFramesNeeded[1][independentVars[1]] #<-- Causing error due to: KeyError: 'Tangible Book Value' .[UPDATE: To solve, incremented index in second operand from 0 to 1. part 2) IT WORKED!] #<-- NOTE: References old version. New version adheres to update. 
        # dataFramesToBeShipped[0][DependentVars[0]] = arrOfDataFramesNeeded[0].loc[arrOfDataFramesNeeded[0].index[0].date().isoformat(),independentVars[0]]/arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[0].index[0].date().isoformat(),independentVars[1]] #<-- Causing error due to: KeyError: 'Tangible Book Value' .[UPDATE: To solve, incremented index in second operand from 0 to 1. part 2) IT WORKED!][UPDATE as of 11/19/25: Error involving key entry. Not sure why, make sure this is focus in next session][part 2) Removing isoformat to see what happens]
        # NOTE: Need a conditional change here, based on PBRatio boolean. 
        pullingFromTickerInfo = True
        if not PRatioBug:
            dataFramesToBeShipped[0].loc[0,DependentVars[0]] = arrOfDataFramesNeeded[0].loc[arrOfDataFramesNeeded[0].index[0].date().isoformat(),independentVars[0]]/arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]] #<-- Causing error due to: KeyError: 'Tangible Book Value' .[UPDATE: To solve, incremented index in second ope rand from 0 to 1. part 2) IT WORKED!][UPDATE as of 11/19/25: Error involving key entry. Not sure why, make sure this is focus in next session]
            print("---")
            print(arrOfDataFramesNeeded[0].columns)
            print("---")
            print(arrOfDataFramesNeeded[0].loc[arrOfDataFramesNeeded[0].index[0].date().isoformat(),independentVars[0]])
            print(arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]])
            dataFramesToBeShipped[0].loc[0,DependentVars[1]] = arrOfDataFramesNeeded[0].loc[arrOfDataFramesNeeded[0].index[0].date().isoformat(),independentVars[2]]/arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[3]]
            dataFramesToBeShipped[0].loc[0,DependentVars[2]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[4]] #<-- Here, NCAV comes from balance sheet. 
            print(dataFramesToBeShipped[0])
            # seriesVersionOn = True <-- This is now a global variable. 
            # end of Assigning the formulas
        elif PRatioBug and pullingFromTickerInfo == True:
            # Need to chang below based on fact that High turns into Shares Issued from arrOfDataFramesNeeded[1] . 
            # print("--DEBUGGING CHECKPOINT: Adressing behavior causing error involving Share Price issue---")
            # pb.set_trace()
            # Decided to replace Price with ticker.info["regularMarketPrice"]
            # dataFramesToBeShipped[0].loc[0,DependentVars[0]] = ticker.info["regularMarketPrice"]/arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]] #<-- Causing error due to: KeyError: 'Tangible Book Value' .[UPDATE: To solve, incremented index in second ope rand from 0 to 1. part 2) IT WORKED!][UPDATE as of 11/19/25: Error involving key entry. Not sure why, make sure this is focus in next session]
            dataFramesToBeShipped[0].loc[0,DependentVars[0]] = ticker.info["priceToBook"] #<-- Causing error due to: KeyError: 'Tangible Book Value' .[UPDATE: To solve, incremented index in second ope rand from 0 to 1. part 2) IT WORKED!][UPDATE as of 11/19/25: Error involving key entry. Not sure why, make sure this is focus in next session]
            print("---")
            print(arrOfDataFramesNeeded[0].columns)
            print("---")
            print(ticker.info["regularMarketPrice"])
            print(arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]])
            dataFramesToBeShipped[0].loc[0,DependentVars[1]] = ticker.info["regularMarketPrice"]/ticker.info["earningsQuarterlyGrowth"]

            
            dataFramesToBeShipped[0].loc[0,DependentVars[2]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[4]] #<-- Here, NCAV comes from balance sheet. 
            # dataFramesToBeShipped[0].loc[0, DependentVars[3]] = ticker.info["regularMarketPrice"]
            print(dataFramesToBeShipped[0])
            print("--DEBUGGING CHECKPOINT: Ensuring that share price returns corrrect value aka ensuring that attrubte is populated properly--")
            # pb.set_trace()
            dataFramesToBeShipped[0].loc[0,"Share Price"] = ticker.info["regularMarketPrice"]
            # seriesVersionOn = True <-- This is now a global variable. 
            # end of Assigning the formulas
        else:
            # Need to chang below based on fact that High turns into Shares Issued from arrOfDataFramesNeeded[1] . 
            print("--DEBUGGING CHECKPOINT: Adressing behavior causing error involving Share Price issue---")
            # pb.set_trace()
            # Decided to replace Price with ticker.info["regularMarketPrice"]
            dataFramesToBeShipped[0].loc[0,DependentVars[0]] = ticker.info["regularMarketPrice"]/arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]] #<-- Causing error due to: KeyError: 'Tangible Book Value' .[UPDATE: To solve, incremented index in second ope rand from 0 to 1. part 2) IT WORKED!][UPDATE as of 11/19/25: Error involving key entry. Not sure why, make sure this is focus in next session]
            print("---")
            print(arrOfDataFramesNeeded[0].columns)
            print("---")
            print(ticker.info["regularMarketPrice"])
            print(arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[1]])
            dataFramesToBeShipped[0].loc[0,DependentVars[1]] = ticker.info["regularMarketPrice"]/arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[3]]

            
            dataFramesToBeShipped[0].loc[0,DependentVars[2]] = arrOfDataFramesNeeded[1].loc[arrOfDataFramesNeeded[1].index[0].date().isoformat(),independentVars[4]] #<-- Here, NCAV comes from balance sheet. 
            print(dataFramesToBeShipped[0])
            # seriesVersionOn = True <-- This is now a global variable. 
            # end of Assigning the formulas
        print("-End of assigning formulas-")
        if(seriesVersionOn):
            dataFramesToBeShipped[0].loc[0, "Company"] = companyAlias
        print("--Entering Debuging Mode for Checkpoint #2--")
        # pb.set_trace() #<-- Will use this to check for vals of variable(s), as well as set certain vars to certain values to cause different behavior.   
        # CHECKPOINT #2: In proof, at this point, formulas will be assigned to neccessary dependent variables for datafrmae that will be output. 
        # At this point, the testSet and trainingSet dataframes will be shipped off. 
        print(dataFramesToBeShipped[0].columns)
        print("--End of Component a[Subsystem 1]--")
        # resultantSeries = dataFramesToBeShipped[0].loc[0, :]
        resultantSeries = dataFramesToBeShipped[0]
        # resultantSeries = <-- will reference a row-tuple involving "P/B", "P/E", "NCAV", "Date For Eval", "company" 
        print("---Debugging checkpoint seeing if P/E key assignment workds for condition #2---")
        # pb.set_trace()
        # NOTE: To solve issue, need to cast dates to only referenced years instead!
        # Body of casting dates to cause assignment to be successful 
        conditionForDataFrame2.index = conditionForDataFrame2.index.year
        balanceSheet.index = balanceSheet.index.year
        # End of Body of casting dates to cause assignment to be successful 
        # NOTE: To solve issue, need to cast dates to only referenced years instead!
        conditionForDataFrame2["P/E"] = conditionForDataFrame2[independentVars[0]]/balanceSheet.loc[:, "Retained Earnings"]
        # Below returns values that are not null, allowing conditional to work properly. FUTURE NOTE: Once rest of pipeline is verified, then may need to change what the PRICE should be. I think it should be share price opposed to high, but we shall see....
        conditionForDataFrame2 = conditionForDataFrame2[conditionForDataFrame2["P/E"].notnull()]
        # Proposed solution: conditionForDataFrame2[conditionForDataFrame2["P/E"].notnull()] test out in next session![UPDATE: This works properly!]
        
        
        newVersion = True
        return (dataFramesToBeShipped if seriesVersionOn == False else resultantSeries) if newVersion == False else (dataFramesToBeShipped if seriesVersionOn == False else [resultantSeries, conditionForDataFrame2])
        # NOTE: compA() is logically correct! Refer to reports for additional steps regarding this!
        # end of component a)
        # component b) [Populating the new dataframe(s) and preparing dataFrames to be written to files[PN: Will mean that resultant dataframes must be copied to nonlocal variables so subsystem1 can access them(UPDATE: Decided to return the array of these dataframes instead)]
        # end of component b)

    # NOTE: Below was replaced by adding system to class's constructor!
    #compB() <-- THis component was abandoned!
class subsys2:
    def __init__(self,param1 = """Insert any params that may be sufficient here!"""):
        print("---Subsystem 2 in Progress---")
        # self.compA()
    # component a)[creating algorithm for assigning optimality for each company's value]
    def compA(self, param1 = """Insert any params that may be sufficient here!""", arrOfCompanies = []): #<-- Not sure if I Need a parameter for this one since I am relying on global variables. AND since its dependent on subsys1. 
        iterators = len(arrOfCompanies) +1 if len(arrOfCompanies) < 1 else len(arrOfCompanies) #2 #<-- Here, just in case for loop is used to iterate through each company. 
        # NOTE: Need to add a condition that ensures that dataframe is updated based on companies passed in. 
        # UPDATE: replacing compA with version where only each company is required. 
        global dataFrameReffingCompanyData
        version0 = False
        if version0 == True:
            for i in range(iterators):
                print("--Component a[Subsystem 2] in progress--")
                # Body of executing algorithm for strategy responsible for assigning optimality[NOTE: This only applies to retrievedDataFrame[0] since that'll be the training set one]
                # Insert psuedosteps here: .
                # predicate wffs for conditions: i) stock(P/B) <= 1.0, ii) stock(P/E) < max(stockPE(P/E,5)), iii) stock(share price) < 0.67*tangible per-share book value[which can be found in historical_data], iv)  (cont here). where stockPE(x,y) = stock's x ratio in the past 5 years and returns z_i, where z_i = x ratio in year i and i \le y .  [predicate wffs written!]
                # NOTE: Share Price == Share Issued and Tangible per-share book value == Tangible Book Value which are both located in the historical_data dataframe. 
                # Body of creating boolExp Array for respective conditions
                arrOfCompanies[i] = param1 #<-- CROSSROADS #1: Here, in the event that I make the iteration take place in the class opposed to main function. 
                # currCompany = "MSFT" or arrOfCompanies[i] #<-- Used default operator here again, thought about replacing None with a param referencing company i.[UPDATE: replaced noe wtih arrOfCompanies[i]] 
                currCompany = arrOfCompanies[i] 
                dataFrameWithCompanyAsAColumnId = subsys1.compA(company=currCompany)[0] #<-- Returns the dataframe referencing company data. [IDEA: THink I need to replace dateInQuestion with companyName. The date should still be included but it shouldn't determine the P/B value. It is only relevant when pulling data]
                # pb.set_trace()
                dataFrameReffingCompanyData = pd.DataFrame() or dataFrameWithCompanyAsAColumnId #<-- REPLACEMENT PENDING: This will be changed to dataframe containing the relevant data. [replacement complete][old version: """retrievedDataFrames[0]""" ][UPDATE: Here, dataFrmaeWithCompanyAsAColumnId refers to the resultant dataframe after assinging rest of columns EXCEPT the optimality]
                dataFrameReffingCompanyData = pd.DataFrame() or dataFrameWithCompanyAsAColumnId #<-- REPLACEMENT PENDING: This will be changed to dataframe containing the relevant data. [replacement complete][old version: """retrievedDataFrames[0]""" ][UPDATE: Here, dataFrmaeWithCompanyAsAColumnId refers to the resultant dataframe after assinging rest of columns EXCEPT the optimality]
                dateInQuestion: datetime = end_date.fromisoformat(end_date) or "" #<-- Using this since vetting process is date-specific. [UPDATE: May not need this variable, but I believe the date should be included as a column with one value?]
                yr = 365;
                yearsRelToDateInQ: datetime = dateInQuestion - timedelta(days=5*yr)
                    # Obtaining stocks highest P/E over previous five years[for assistance, use this search query: `? does ticker object have a start date parameter`]
                    # end of Obtaining Stocks highest P/E over previous five years
                # arrForCondTwo: pd.DataFrame = arrOfDataFramesNeeded[len(arrOfDataFramesNeeded) - 1] #<-- Did this since I added the dataframe specifically for cond 2 at the END of arrOfDataFramesNeeded. 
                arrForCondTwo: pd.DataFrame = arrOfDataFramesNeeded[len(arrOfDataFramesNeeded) - 1] #<-- Did this since I added the dataframe specifically for cond 2 at the END of arrOfDataFramesNeeded. [UPDATE: Need to replace this]
            #    return #<-- Testing version of return statement[anything above this return statement is sucessful and everything below hasn't been tested yet. THis is relative to each function][for subsys2(compA())][
                highestP_EOvrFiveYrs: float = 5 or arrForCondTwo["P/E"].max() #<-- REPLACEMENT PENDING: will be replaced when above is filled in. [UPDATE: Using default operator logic to set val to trusy val if ither operand is undef. Will involve querying dataframe for maximum val in column]
                # POTENTIAL IMPROVEMENT PENDING: Need to work on modifying bool exps below by ensuring that they reference the dep var attribute for that PARTICULAR day in which decision must be made. 
                #highestP_EOvrFiveYrs = 
                # UPDATE: THe inclusion of ternary operations below allows me to create env for determining optimality based on the day these decisions are being made. 
                # NOTE: Below will reference command for replacing dateInQuestion with currCompany: 
                # . `157m a; `.,'as/dateInQ\w\+/currCompany/g`  by running this on line of boolExps. [Replaced, in the case where multiply companies are iterated. 
                boolExps: list[bool] = [
                        dataFrameReffingCompanyData["P/B" if currCompany == None else ("P/B",currCompany)] <= 1.0,
                    dataFrameReffingCompanyData["P/E" if currCompany == None else ("P/E",currCompany)] <= 0.4*highestP_EOvrFiveYrs,
                    dataFrameReffingCompanyData["Share Price" if currCompany == None else ("Share Price",currCompany)] < 0.67*dataFrameReffingCompanyData["Tangible Book Value" if currCompany == None else ("Tangible Book Value",currCompany)]
                ]
                    # End of Body of creating boolExp Array for respective conditions
                score = 0
                if boolExps[0]:
                    score += 1
                if boolExps[1]:
                    score += 1
                if boolExps[2]:
                    score += 1
                #scoreTable[f"{currCompany}"] = score; #<-- Will be used later to assign optimality [UPDATE: May not be needed, need to make sure it is added to optimality col for comapny name[also, will make sense to have companies be the index!]
                # dataFrameReffingCompanyData[i, "company"] = f"{currCompany}"; # NOTE: This is under assumption that this refers to row tuple i and assigns company to that tuple. DEBUGGING OPPORTUNITY #1: May need to set a breakpoint/pdb.set_trace command below this! [UPDATE: This is not needed anymore since subsys1(compA) covers this already!]
                dataFrameReffingCompanyData["Optimality"] = score; #<-- CROSSROADS OPPORTUNITY: a) This process works on assumption that each company is iteratively having the optimality algorithm applied to it. 
                # End of Body of executing algorithm for strategy responsible for assigning optimality
               # NOTE: This component will have to take in a company as a parameter. 
        else:
            # Version of code where only company[i] is proprioritezed and a series is returned with optimality there. 
            print("--Component a[Subsystem 2] in progress--[VERSION #2]")
            # Body of executing algorithm for strategy responsible for assigning optimality[NOTE: This only applies to retrievedDataFrame[0] since that'll be the training set one]
            # Insert psuedosteps here: .
            # predicate wffs for conditions: i) stock(P/B) <= 1.0, ii) stock(P/E) < max(stockPE(P/E,5)), iii) stock(share price) < 0.67*tangible per-share book value[which can be found in historical_data], iv)  (cont here). where stockPE(x,y) = stock's x ratio in the past 5 years and returns z_i, where z_i = x ratio in year i and i \le y .  [predicate wffs written!]
            # NOTE: Share Price == Share Issued and Tangible per-share book value == Tangible Book Value which are both located in the historical_data dataframe. 
            # Body of creating boolExp Array for respective conditions
            # arrOfCompanies[i] = param1 #<-- CROSSROADS #1: Here, in the event that I make the iteration take place in the class opposed to main function. [UPDATE: Decided to have iteration take place in the main function opposed to this function]
            # currCompany = "MSFT" or arrOfCompanies[i] #<-- Used default operator here again, thought about replacing None with a param referencing company i.[UPDATE: replaced noe wtih arrOfCompanies[i]] 
            currCompany = param1
            # dataFrameWithCompanyAsAColumnId = subsys1.compA(company=currCompany)[0] #<-- Returns the dataframe referencing company data. [IDEA: THink I need to replace dateInQuestion with companyName. The date should still be included but it shouldn't determine the P/B value. It is only relevant when pulling data] <-- UPDATE: THis may not be needed. 
            # dataFrameReffingCompanyData = pd.Series() or dataFrameWithCompanyAsAColumnId #<-- REPLACEMENT PENDING: This will be changed to dataframe containing the relevant data. [replacement complete][old version: """retrievedDataFrames[0]""" ][UPDATE: Here, dataFrmaeWithCompanyAsAColumnId refers to the resultant dataframe after assinging rest of columns EXCEPT the optimality]
            # dataFrameWithCompanyAsAColumnId = subsys1().compA(company=currCompany) #<-- Returns the dataframe referencing company data. [IDEA: THink I need to replace dateInQuestion with companyName. The date should still be included but it shouldn't determine the P/B value. It is only relevant when pulling data] [UPDATE: Trying to have compA(subsys1) return seriess AND the dataframe used for conditions!]
            """
            dataFrameWithCompanyAsAColumnId = subsys1().compA(company=currCompany) #<-- Returns the dataframe referencing company data. [IDEA: THink I need to replace dateInQuestion with companyName. The date should still be included but it shouldn't determine the P/B value. It is only relevant when pulling data]
            dataFrameReffingCompanyData = dataFrameWithCompanyAsAColumnId
            # Body of version where compA(subsys1) returns TWO things[uncomment below once version is established!] 
            """
            # pb.set_trace()
            arrOfDataFramesNeeded = subsys1().compA(company=currCompany) 
            dataFrameWithCompanyAsAColumnId = arrOfDataFramesNeeded[0] #<-- Returns the dataframe referencing company data. [IDEA: THink I need to replace dateInQuestion with companyName. The date should still be included but it shouldn't determine the P/B value. It is only relevant when pulling data]
            dataFrameReffingCompanyData = arrOfDataFramesNeeded[1]
            arrForCondTwo = arrOfDataFramesNeeded[1]            
            
            
            # Body of version where compA(subsys1) returns TWO things
            # pb.set_trace()
            dateInQuestion: datetime = datetime.fromisoformat(end_date) or "" #<-- Using this since vetting process is date-specific. [UPDATE: May not need this variable, but I believe the date should be included as a column with one value?]
            yr = 365;
            yearsRelToDateInQ: datetime = dateInQuestion - timedelta(days=5*yr)
                # Obtaining stocks highest P/E over previous five years[for assistance, use this search query: `? does ticker object have a start date parameter`]
                # end of Obtaining Stocks highest P/E over previous five years
            # arrForCondTwo: pd.DataFrame = arrOfDataFramesNeeded[len(arrOfDataFramesNeeded) - 1] #<-- Did this since I added the dataframe specifically for cond 2 at the END of arrOfDataFramesNeeded. 
            # ticker = yf.Ticker(f"{currCompany}") #<-- NOTE: Will need to replace this with multiple tickers later on.            
            # arrForCondTwo: pd.DataFrame = ticker.history(period="5y").tz_localize(None)
            # arrForCondTwo: pd.DataFrame = [<-- may or may not need to use this again. Mentioned it a few lines above] 
            #    return #<-- Testing version of return statement[anything above this return statement is sucessful and everything below hasn't been tested yet. THis is relative to each function][for subsys2(compA())][
            # highestP_EOvrFiveYrs: float = 5 or arrForCondTwo["P/E"].max() #<-- REPLACEMENT PENDING: will be replaced when above is filled in. [UPDATE: Using default operator logic to set val to trusy val if ither operand is undef. Will involve querying dataframe for maximum val in column]
            # print("---Debugging Checkpoint: Checking if P/E is a key in arrForCOndTwo")
            # pb.set_trace() <-- NOTE: P/E solution works properly!
            highestP_EOvrFiveYrs = arrForCondTwo["P/E"].max() #<-- REPLACEMENT PENDING: will be replaced when above is filled in. [UPDATE: Using default operator logic to set val to trusy val if ither operand is undef. Will involve querying dataframe for maximum val in column]
            # POTENTIAL IMPROVEMENT PENDING: Need to work on modifying bool exps below by ensuring that they reference the dep var attribute for that PARTICULAR day in which decision must be made. 
            #highestP_EOvrFiveYrs = 
            # UPDATE: THe inclusion of ternary operations below allows me to create env for determining optimality based on the day these decisions are being made. 
            # NOTE: Below will reference command for replacing dateInQuestion with currCompany: 
            # . `157m a; `.,'as/dateInQ\w\+/currCompany/g`  by running this on line of boolExps. [Replaced, in the case where multiply companies are iterated. 
            """ boolExps: list[bool] = [
                    dataFrameReffingCompanyData["P/B" if currCompany == None else ("P/B",currCompany)] <= 1.0,
                dataFrameReffingCompanyData["P/E" if currCompany == None else ("P/E",currCompany)] <= 0.4*highestP_EOvrFiveYrs,
                dataFrameReffingCompanyData["Share Price" if currCompany == None else ("Share Price",currCompany)] < 0.67*dataFrameReffingCompanyData["Tangible Book Value" if currCompany == None else ("Tangible Book Value",currCompany)]
                ]"""
                # NOTE: No errors occur until up to THIS point. Following error is prod by the boolExps: pandas.errors.IndexingError: Too many indexers
            """             boolExps: list[bool] = [
                    dataFrameReffingCompanyData.loc["P/B" if currCompany == None else ("P/B",currCompany)] <= 1.0,
                dataFrameReffingCompanyData.loc["P/E" if currCompany == None else ("P/E",currCompany)] <= 0.4*highestP_EOvrFiveYrs,
                dataFrameReffingCompanyData.loc["Share Price" if currCompany == None else ("Share Price",currCompany)] < 0.67*dataFrameReffingCompanyData["Tangible Book Value" if currCompany == None else ("Tangible Book Value",currCompany)]
            ]
            """ 
            print("---DEBUGGING SUBCHECKPOINT: CHecking for the Boolean Expressions---")
            # pb.set_trace()
            # NOTE: dataFrameReffingCompanyData is a pd.Series so the querying used Above is NOT needed!
            # dataFrameThatRefsSharePrice = pd.DataFrame() #<-- This variable is self explanatory. Need to modify soon. 
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
            """             
            if boolExps[0]:
                score += 1
            if boolExps[1]:
                score += 1
            if boolExps[2]:
                score += 1
            """            
            # `N`OTE: May need to change some things in future...NOT sure[particularly, involving ][NOTE: Will need to work on validating boolean exps since all optimality cols are equal to 0]
            score = 0
            if boolExps[0].any():
                score += 1
            if boolExps[1].any():
                score += 1
            if boolExps[2].any():
                score += 1
            #scoreTable[f"{currCompany}"] = score; #<-- Will be used later to assign optimality [UPDATE: May not be needed, need to make sure it is added to optimality col for comapny name[also, will make sense to have companies be the index!]
            # dataFrameReffingCompanyData[i, "company"] = f"{currCompany}"; # NOTE: This is under assumption that this refers to row tuple i and assigns company to that tuple. DEBUGGING OPPORTUNITY #1: May need to set a breakpoint/pdb.set_trace command below this![UPDATE as of 11/21/25: May NOT need this at all!]
            # print("---Testing if the optimality actually works[at least the assignments]---")
            # pb.set_trace()
            # dataFrameReffingCompanyData[f"{currCompany}", "Optimality"] = score; #<-- CROSSROADS OPPORTUNITY: a) This process works on assumption that each company is iteratively having the optimality algorithm applied to it. [UPDATE: Not needed anymore since series are being returned opposed to entire dataframe] 
            # Below's verison references the pd.Series() so .loc isn't needed here. 
            dataFrameWithCompanyAsAColumnId["Optimality"] = score; # UPDATE: Works as intended. Focu sshould shift back towards ensuring that Share Price thing works properly. 
            # pb.set_trace()
            # End of Body of executing algorithm for strategy responsible for assigning optimality
            # NOTE[EXTREMELY IMPORTANT]: Succeeding this point, another component will be responsible for combining the resultant dataframes, ordering the companies by optimality, and then assigning ranks to the companies, and then finishing off my adding the rank numbers to the resultant dataframe. 
            # NOTE: This component will have to take in a company as a parameter. 

            print("--End of Component a[Subsystem 2] in progress--")
            return dataFrameWithCompanyAsAColumnId
# end of component a)
    # component b)
    def compB(): # NOTE: I believe compB(subsystem2) is complete! 
        print("--Component b[Subsystem 2] in progress--")
        global retreivedDataFrames 
        #retreivedDataFrames = dataFramesReffingCompanyData or pd.DataFrame()#compA() #<-- Will be replaced soon[REPLACEMENT PENDING][UPDATE: Replaced with global var referencign resultant dataframe(s) with optimality columns]. 
        retreivedDataFrames = [dataFrameReffingCompanyData,dataFrameReffingCompanyData] or pd.DataFrame()#compA() #<-- Will be replaced soon[REPLACEMENT PENDING][UPDATE: Replaced with global var referencign resultant dataframe(s) with optimality columns]. [UPDATE: Used an array of two insts of Company Data, since one of them will be used for trainingSet and testSet respectively(assuming it isn't done in Model Dev file)]
        # Body of writing resp dataframes to files[need to use to_csv I believe]
        filePathToTrainingSetDir: str = "MLLifecycle/ModelDevelopment/TrainingSets" #<-- fill in later[complete]
        filePathToTestSetDir: str = "MLLifecycle/ModelDevelopment/TestingSets" #<-- fill in later[complete]
        print("--End of Component b[Subsystem 2]--")
        print("Entering Debuging Mode for Checkpoint #3")
        # pb.set_trace() #<-- Will use this to check for vals of variable(s), as well as set certain vars to certain values to cause different behavior.   
        # CHECKPOINT #3: In proof, at this point, dataframes will be written to the neccessary files to be ingested by the Model.  
        print("--Writing Dataframes for Model Ingestion--")
        retreivedDataFrames[0].to_csv(f'{filePathToTrainingSetDir}/trainingSet{setNum if setNum != None else 1}')
        retreivedDataFrames[1].to_csv(f'{filePathToTrainingSetDir}/testSet{setNum if setNum != None else 1}')
        # End of Body of writing resp dataframes to files
        print("--End of Writing Dataframes for Model Ingestion--")
        print("--End of Component b[Subsystem 2]--")
        return #<-- Testing version of return statement[anything above this return statement is sucessful and everything below hasn't been tested yet. THis is relative to each function]
    # end of component b)

    # NOTE: Below was replaced by adding system to class's constructor!
    #return
    #compA() 
    #compB()












def main(companies: list[str] = []):
    print("---Starting Data Prep Process---")

    companies: list[str] = ["GOOG","AAPL", "AMZN", "MSFT"] if companies == None else companies
    
    listOfSeriesToCreateDataFrame = []
    for i in range(len(companies)):
        print(f"----Adding company {companies[i]} to engineered dataset----")
        # """
        # NOTE: Will uncomment, once everything with the functions used here is situated[add a checklist here: ]
        print("---Starting Subsystem 1---")
        # pb.set_trace(); #<-- Adding breakpoint here to see what happens. 
        # Call function referencing topmost subsystem #1 here: 
        a: str = ""; b: str = ""
        # subsys1(param1=company_i)
        # subsys1().compA(company=company_i)
        # resultantDataFrame = pd.concat([resultantDataFrame, subsys1().compA(company=companies[i])])
        # resultantDataFrame.loc[i, ["P/B", "P/E", "NCAV", "Company"]] = subsys1().compA(company=companies[i])
        subsys1().compA(company=companies[i])
        # companySeries= subsys1().compA(company=companies[i])
        # pb.set_trace()
        # subsys2().compA(param1=companySeries)
        subsys2().compA(param1=companies[i])
        retSeries = True
        if retSeries:
            # ^^ NOTE: Above is returning a series each time[at least this is the assumption]
            companySeries = subsys2().compA(param1=companies[i])
            listOfSeriesToCreateDataFrame.append(companySeries)
        else:
            # ^^ NOTE: Above is NOT retruning a series
            print("---Assumption that series is NOT returned---")
            # NOTE: Not sure what to put here. Will default to returning series from compA(subsys2)
            # companySeries = subsys2().compA(param1=companies[i])
            # listOfSeriesToCreateDataFrame.append(companySeries)
            
        # resultantDataFrame.loc[i, :] = subsys1().compA(company=companies[i]) #<-- This line is still causing problems. 
        # Will replace above with this: subsys1(company_i)
        
        print("---End of Subsystem 1---")
        
        """
        print("---Starting Subsystem 2---")
        # Call function referencing topmost subsystem #2 here: 
        # subsys2(b)
        # subsys2.compA(b) #<-- Here, need to ensure that 
        # Will replace above with this: subsys2(company_i)
        print("---End of Subsystem 2---")
        """
        print(f"----End of Adding company {companies[i]} to engineered dataset----")

    pb.set_trace()
    # resultantDataFrame = pd.DataFrame(listOfSeriesToCreateDataFrame) #<-- Causing following error: *** ValueError: Must pass 2-d input. shape=(1, 1, 6)[need to figure out how to resolve error][UPDATE: Error fixed! Imp below resolved issue. NOW: Main focus goes back to setting up optimality!]
    resultantDataFrame = pd.concat([pd.DataFrame(x) for x in listOfSeriesToCreateDataFrame]).reset_index() #<-- used list comprehension to transform listOfSeries to resultantDataFrame. 
    del resultantDataFrame['index']
    print(resultantDataFrame) #<-- THis dataframe will reference the dataframe that adheres to the follwowing object: company(CompanyName, "P/B", "P/E", "NCAV", "Date For Eval", "Optimality", (cont here if applicable))[NOTE: Will be wise to make a Entity via ERDs for documentation when writing paper at end]
    inNoteBook = False
    filePathToModelDir = "C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/projectCode/MLLifecycle/ModelDevelopment/preparedDataset.csv" if inNoteBook == False else "preparedDataset.csv"
    # Body of handling edge case where all of them are same optimality

    resultantDataFrame.to_csv(f"{filePathToModelDir}")
    if((resultantDataFrame["Optimality"] == 0).all() == True):
        # Setting optimality column to be based on alphabetical ordering 
        print("---DEBUGGING CHECKPOINT #3: Validating process of creating resultant dataframe and sorting columns and assigning optimality, iff all stocks chosen all have same optimality---")
        pb.set_trace()
        # UPDATE: Below works as expected!
        resultantDataFrame.sort_values(by='Company',inplace=True)
        resultantDataFrame = resultantDataFrame.set_index(np.arange(4))
        resultantDataFrame.loc[:,"Optimality"] = pd.Series(np.arange(resultantDataFrame["Optimality"].shape[0]))

        # End of Setting optimality column to be based on alphabetical ordering 
        resultantDataFrame.to_csv(f"{filePathToModelDir}")
    
     
    
    
    # End of Body of handling edge case where all of them are same optimality


# NOTE: THE DATA ENGINEERING IS COMPLETE. Need to now copy and paste this version of code!
main()
