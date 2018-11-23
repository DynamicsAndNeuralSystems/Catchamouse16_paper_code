library(tsfeatures)

# go to op_importance directory
setwd("/Users/carl/PycharmProjects/op_importance/")

# list files in data dir
dataDir <- "UCR2018_CSV"

for (file in list.files(dataDir)){
	
	if (grepl("_class", file, fixed=TRUE)){
		next
	}
	if (grepl("done", file, fixed=TRUE)){
		next
	}

	fileLoc <- paste(dataDir, "/", file, sep="")

	print(fileLoc)

	# read in the data from csv
	MyData <- read.csv(file=fileLoc, header=FALSE, sep=",")

	myfeatures <- tsfeatures(list(t(MyData[1,])))
	
	for (i in 2:dim(MyData)[1]){
		print(i)
		
		tryCatch({myfeaturesTemp <- tsfeatures(list(t(MyData[i,])))}, error = function(e) {myfeaturesTemp <- rep(NA, 16)})
		
		myfeatures = rbind(myfeatures, myfeaturesTemp)
	}

	write.csv(myfeatures, file = paste("tsfeatures_out/", file, sep=""))
}




