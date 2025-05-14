require('raster')
require('rgdal')
library(INLA)
library(RColorBrewer)
library(zoo) 
library(scales)
library(matrixStats)
ll.to.xyz<-function(ll){
	if(is.null(colnames(ll))){
		colnames(ll)<-c('longitude','latitude')	
	}
	if(colnames(ll)[1]=='x' & colnames(ll)[2]=='y'){
		colnames(ll)<-c('longitude','latitude')
	}
	if(colnames(ll)[1]=='lon' & colnames(ll)[2]=='lat'){
		colnames(ll)<-c('longitude','latitude')
	}

	ll[,'longitude']<-ll[,'longitude']*(pi/180)
	ll[,'latitude']<-ll[,'latitude']*(pi/180)
	
	x = cos(ll[,'latitude']) * cos(ll[,'longitude'])
	
	y = cos(ll[,'latitude']) * sin(ll[,'longitude'])
	
	z = sin(ll[,'latitude'])

	return(cbind(x,y,z))
}  
match.cols<-function(val){
    n=1000
    col<-data.frame(val=seq(min(val),max(val),length.out=n),col=colfunc(n))
    out<-rep(NA,length(col))
    for(i in 1:length(val)){
        out[i]<-as.character(col[which.min(abs(col$val-val[i])),'col'])
    }	
    return(out)
}
bias=1
colfunc <- colorRampPalette(c('yellow','orange','red','brown'),bias=bias)
emplogit<-function(Y,N){
	top=Y*N+0.5
	bottom=N*(1-Y)+0.5
	return(log(top/bottom))
}
colfunc <- colorRampPalette(c(
	rgb(0/255,0/255,255/255),
	rgb(43/255,65/255,255/255),
	rgb(56/255,109/255,255/255),
	rgb(59/255,157/255,255/255),
	rgb(48/255,207/255,255/255),
	rgb(0/255,255/255,255/255),
	rgb(112/255,255/255,210/255),
	rgb(161/255,255/255,164/255),
	rgb(199/255,255/255,120/255),
	rgb(231/255,255/255,74/255),
	rgb(255/255,255/255,0/255),
	rgb(255/255,213/255,0/255),
	rgb(255/255,166/255,0/255),
	rgb(255/255,123/255,0/255),
	rgb(255/255,77/255,0/255),
	rgb(255/255,0/255,0/255)
	),bias=bias)

nfeat = 1024
threshold_itn = 0.1
threshold_act = 0.5
threshold_year = 2008

# load pfpr dataset and process it
d=readRDS('/home/pfpr.rds')
d=d[!is.na(d$PfPr),] # remove NA prevalence

d$PfPr_logit<-emplogit(d$PfPr,d$Nexamined) #add empirical logit

d=d[d$itnavg4<threshold_itn & 
	d$act<threshold_act  & 
	d$irs<threshold_itn & 
	d$yearqtr<threshold_year,] #subset data

design_matrix = d[,paste0('feature_',0:(nfeat-1))] # get features from contrastive,nfeat-1 for python 0 indexing
colnames(design_matrix)<-paste0('V',1:ncol(design_matrix)) # rename them


# load prediction csv and rename them
pred=readRDS('/home/pred.rds')
pred=pred[complete.cases(pred),]
design_matrixp = pred[,paste0('feature_',0:(nfeat-1))] # nfeat-1 for python 0 indexing
colnames(design_matrixp)<-paste0('V',1:ncol(design_matrixp))

betas = read.csv('/home/coeffs_0_1.csv',header=F) # load coefficients from numpyro, intercept and coefs
betas_mean = as.numeric(colMeans(betas)) # standard mean

lp_mean = betas_mean[1] + as.matrix(design_matrixp)%*%cbind(betas_mean[2:(nfeat+1)]) # 1025 features with intercept
plot(pred$lon,pred$lat,col=match.cols(plogis(as.vector(lp_mean))),pch=16,cex=0.1,xlab='Lon',ylab='Lat')

admin <- raster('/home/admin2023_0_MG_5K.tif')
funding = read.csv('/home/Africa_funding_2005_2022_MAP.csv')
population = raster('/home/ihme_corrected_worldpop_All_Ages_3_2023.tif')
funding[funding=='-']=NA
funding$pfpr = NA
NAvalue(admin) <- -9999
NAvalue(population) <- -9999
agg_population = aggregate(population,2,sum)
population = extract(agg_population, cbind(pred$lon, pred$lat))


admin_codes <- extract(admin, cbind(pred$lon, pred$lat))
unique_admin_codes <- unique(admin_codes)
unique_admin_codes = unique_admin_codes[!is.na(unique_admin_codes)]
lp_admin <- unique_admin_codes
for (i in 1:length(unique_admin_codes)) {
	pop = population[admin_codes == unique_admin_codes[i]]
	funding$pop = sum(pop,na.rm=T)
	pop = pop/sum(pop,na.rm=T)
	mn =  sum(plogis(lp_mean[admin_codes == unique_admin_codes[i]])*pop, na.rm = TRUE)
	lp_admin[i] <- mn
	funding$pfpr[funding$Raster_Value==unique_admin_codes[i]]=mn
}

num = as.matrix(funding[,paste0('X',2005:2022)])
num <- apply(num, 2, as.numeric)
funding$total = rowSums(num,na.rm=T)

a=data.frame(x = qlogis(funding$pfpr),y =(funding$total/funding$pop))
summary(lm(y~x,a))

a=data.frame(x = qlogis(funding$hdi),y =log(funding$total))
summary(lm(y~x,a))


funding$binary_hdi = funding$hdi
funding$binary_hdi[funding$hdi>=median(funding$hdi)]=1
funding$binary_hdi[funding$hdi<=median(funding$hdi)]=0

a=data.frame(binary_hdi =funding$binary_hdi, hdi = funding$hdi,pfpr = funding$pfpr,y =log(funding$total))
summary(lm(y~hdi + pfpr,a))

a=data.frame(binary_hdi =funding$binary_hdi, hdi = funding$hdi,pfpr = funding$pfpr,y =log(funding$total))
summary(lm(y~binary_hdi + pfpr,a))

####### realisations


makeraster=function(l){
	r=raster('/home/statelite/master_10km_raster.tif');NAvalue(r)=-9999
	r[!is.na(r)]=0
	cells = cellFromXY(r,cbind(pred$lon,pred$lat))
	wh = !is.na(cells)
	r[r==1]=0
	r[cells] = plogis(l[wh])
	r[r>2]= NA
	return(r)
}

library(doParallel)
registerDoParallel(20)
lp_sample <- foreach(i = 1:nrow(betas),.combine=cbind) %dopar% {
	c1 = as.numeric(betas[i,])
	return(c1[1] + as.matrix(design_matrixp)%*%cbind(c1[2:(nfeat+1)]))
}

pfpr_stack = stack(makeraster(lp_sample[,1]))
for(i in 2:nrow(betas)){
	pfpr_stack[[i]] = makeraster(lp_sample[,i])
}

incidence_stack = stack(paste0('/home/extracted_folder/realisation.',1:200,'.inc.rate.all.tif'))



library(progress)
pb <- progress_bar$new(total = ncol(lp_sample))
admin <- raster('/home/admin2023_1_MG_5K.tif')
limits=raster('/home/statelite/pf_limits.tif',NAflag=-9999)
limits[!is.na(limits)]=1

NAvalue(admin) <- -9999
admin_codes <- extract(admin, cbind(pred$lon, pred$lat))
unique_admin_codes <- unique(admin_codes)
unique_admin_codes = unique_admin_codes[!is.na(unique_admin_codes)]
lp_admin <- lp_sample

for (j in 1:ncol(lp_sample)) {
  for (i in 1:length(unique_admin_codes)) {
  	pop = population[admin_codes == unique_admin_codes[i]]
  	pop = pop/sum(pop,na.rm=T)
	pr = lp_sample[admin_codes == unique_admin_codes[i], j]
    lp_admin[admin_codes == unique_admin_codes[i], j] <- sum(pop*pr, na.rm = TRUE)
  }
  pb$tick()
}


l=rowQuantiles(lp_admin,probs=c(0.025))
h=rowQuantiles(lp_admin,probs=c(0.975))
m=rowQuantiles(lp_admin,probs=c(0.5))

par(mfrow=c(1,3))
plot(pred$lon,pred$lat,col=match.cols(plogis(as.vector(l))),pch=16,cex=0.1,xlab='Lon',ylab='Lat')
plot(pred$lon,pred$lat,col=match.cols(plogis(as.vector(m))),pch=16,cex=0.1,xlab='Lon',ylab='Lat')
plot(pred$lon,pred$lat,col=match.cols(plogis(as.vector(h))),pch=16,cex=0.1,xlab='Lon',ylab='Lat')

makeraster=function(l){
	r=raster('/home/statelite/master_10km_raster.tif');NAvalue(r)=-9999
	r[!is.na(r)]=0
	cells = cellFromXY(r,cbind(pred$lon,pred$lat))
	wh = !is.na(cells)
	r[r==1]=0
	r[cells] = plogis(l[wh])
	r[r>2]= NA
	return(r)
}

library(rnaturalearth)
raster_data=makeraster(m)
countries <- ne_countries(scale = "medium", returnclass = "sf")

image.plot(raster_data, legend.width = 1.5, legend.mar = 5, col = rev(terrain.colors(1000)), zlim = c(0, 1))
plot(countries, add = TRUE, border = "black", col = NA)


plot(lp,d$PfPr_logit,pch=16,cex=0.5,col=alpha('black',0.5),xlab='Predicted',ylab='Observed')
abline(0,1,col='red')


admin = raster('/home/african_admin1_5km_2013_a0_match.tif');NAvalue(admin)=-9999
ad = extract(admin,cbind(pred$lon,pred$lat))
un = unique(ad)
lpa = rep(0,length(lpp))
for(i in 1:length(un)){
	lpa[ad==un[i]] = mean(lpp[ad==un[i]],na.rm=T)
}
plot(pred$lon,pred$lat,col=match.cols(plogis(as.vector(lpa))),pch=16,cex=0.1,xlab='Lon',ylab='Lat')


r=raster('/home/statelite/master_10km_raster.tif');NAvalue(r)=-9999
r3=r


r3[!is.na(r3)]=0
cells = cellFromXY(r,cbind(pred$lon,pred$lat))
wh = !is.na(cells)
r3[r3==1]=0
r3[cells] = plogis(lpa[wh])
r3[r3>2]= NA

limits=raster('/home/statelite/pf_limits.tif',NAflag=-9999)
r2 <- resample(limits, r3, method = "bilinear")
r2[!is.na(r2)]=1
# Mask raster1 with raster2
pfpr <- mask(r3, r2)

plot(pfpr,col=colfunc(10))

r=raster('/home/statelite/master_10km_raster.tif');NAvalue(r)=-9999
r3=r
r3[!is.na(r3)]=0
cells = cellFromXY(r,cbind(pred$lon,pred$lat))
wh = !is.na(cells)
r3[r3==1]=0
r3[cells] = plogis(lpp[wh])
r3[r3>2]= NA

limits=raster('/home/statelite/pf_limits.tif',NAflag=-9999)
r2 <- resample(limits, r3, method = "bilinear")
r2[!is.na(r2)]=1
# Mask raster1 with raster2
pfpr <- mask(r3, r2)

plot(pfpr,col=colfunc(10))


##########

library(raster)

s = c(
'/home/ts/TSI-Martens2-Pf.202101.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202102.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202103.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202104.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202105.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202106.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202107.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202108.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202109.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202110.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202111.Data.5km.Data.tif',
'/home/ts/TSI-Martens2-Pf.202112.Data.5km.Data.tif')


st=stack(s);NAvalue(st)=-9999
m = calc(st,mean)

pop = raster('/home/ihme_corrected_worldpop_All_Ages_3_2023.tif');NAvalue(pop)=-9999

m=crop(m,pop)

v1 = getValues(m)
v2 = getValues(pop)

wh = is.na(v1) | is.na(v2)

v1=v1[!wh]
v2=v2[!wh]
N=sum(v2)

breaks <- seq(0, 1, by = 0.1)
b <- cut(v1, breaks = breaks, include.lowest = TRUE)
un =unique(b)
prop = rep(0, length(un))
for(i in 1:length(un)){
	prop[i] = sum(v2[b==un[i]])/N

}

x = breaks-0.05
x = x[-1]
library(ggplot2)
# Example data
data <- data.frame(
  Category = x,
  Value = prop
)

# Create the plot
p <- ggplot(data, aes(x = Category, y = Value)) +
  geom_bar(stat = "identity") # Use stat = "identity" for bar heights to represent actual values



x = breaks-0.05
x = x[-1]

plot(x,100*cumsum(prop),pch=16,type='b',xlab='Max Temperature Suitability',ylab="Cumulative % of population")
