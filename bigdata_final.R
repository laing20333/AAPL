
#### import data and sort
data<-read.csv("table.csv")
data_new <- order(data$Date)
data_new1 <- data[data_new,]
rownames(data_new1) <- c()


####calculate Return
Return <- c()
for( i in 1:nrow(data_new1) ) {
  Return <- c( Return, (data_new1[i+1,5]-data_new1[i,5])/data_new1[i,5] )
}

#### data with Return
data_new2 <- data.frame( data_new1 , Return )


#### calculate garch(1, 1)
## data
Return = data_new2$Return


#_> setting parameter
L.R.variance <- 0.0035/100
gamma <- 5/100
alpha <- 1/100
beta <- 80/100


#_> function
open_garch11 <- function(u, s) {
    sigman <- L.R.variance*gamma + alpha*u^2 + beta*s 
}

#_> main 

vol = c() #volatility
for ( i in 1:length(Return) ){
    if (i == 1){
        vol = c(vol, open_garch11( Return[1], var(Return, na.rm=TRUE)^2 ))
    }else{
        vol = c(vol, open_garch11( Return[i], vol[i-1] ))
    }    
}


data_ALL <- data.frame(data_new2, Volatility = vol)


#### get {Return, Volatility, Close} data on iPhone unveil +-30days 
attach(data_ALL)
head(data_ALL)
###get the index of launch day
launch_day <- c("2007-01-09", "2008-06-09", "2009-06-08", "2010-06-07",
                "2011-10-04", "2012-09-12", "2013-09-10", "2014-09-09")

for(i in 2007:2014) { 
    #name_lanuchday <- paste("index",i, sep="")
    name_Return <- paste("Return",i, sep="")
    name_Volatility <- paste("Volatility",i, sep="")
    name_Close <- paste("Close",i, sep="")
    
    
    #start(a) end(b)
    a <- match(launch_day[i - 2006], Date) - 30
    b <- match(launch_day[i - 2006], Date) + 30
    
    
    #date 61 days
    Return61<- Return[a:b]*10
    Volatility61<- Volatility[a:b]*10^(5)
    Close61<- Close[a:b]/100
    
    
    # formate data to comma-separated 
    R61 <- c()
    for ( i in 1:61) {
        R61 <-paste(R61, Return61[i],",", sep = "") 
    }
    
    
    V61 <- c()
    for ( i in 1:61) {
        V61 <-paste(V61, Volatility61[i],",", sep = "") 
    }
    
    C61 <- c()
    for ( i in 1:61) {
        C61 <-paste(C61, Close61[i],",", sep = "") 
    }
    
    assign(name_Return, print(R61) )
    assign(name_Volatility, print(V61) )
    assign(name_Close, print(C61))
    
}
detach()
