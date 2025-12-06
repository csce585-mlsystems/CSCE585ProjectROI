# Purpose: This file will be responsible for producing all the plots that come about in the experiment. 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
# Body of sequentially outputing the plots in question
module_dir3 = os.path.join('C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/plots')
sys.path.insert(0,module_dir3)

module_dir4 = os.path.join('/c/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/projectCode/MLLifecycle/ModelDevelopmentAndTraining/plots')
sys.path.insert(1,module_dir4)
numOfExperiments = 3 #<-- update this if neccessary
numOfImages = 2*numOfExperiments #<-- update this later on
listOfFilePathsReffingPlots = [{"filePath": f"./plots/Model{'Accuracy' if x % 2 == 0 else 'Loss'}#{1 if x % 2 == 0 else 2}"} for x in range(numOfImages)]

# NOTE: For this to work, will have to add code in model file responsible for writing plot
# to a file!
for i in range(len(listOfFilePathsReffingPlots)):
    # img = mpimg.imread(image_path)
    img = mpimg.imread(listOfFilePathsReffingPlots[i]["filePath"])
    plt.imshow(img)
    # Line of code that writes image to a file in directory to be seen by viewers


# End of Body of sequentially outputing the plots in question
