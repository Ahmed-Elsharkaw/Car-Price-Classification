# Car-Price-Classification 
using Xgboost to predict the price categrory with complete preprocessing and feature selection and feature engineering
The objective of the project is to prepare to apply different machine
learning algorithms to real-world tasks. This will help the you to increase their
knowledge about the workflow of the machine learning tasks. you will learn
how to clean data, apply pre-processing, feature engineering, regression, and
classification techniques. The project will consist of two competitions that will be
delivered in milestones.



Dropping segment and color columns
Filling Nan values by mode in Drive unit column
Filling Nan values by mean in Volume column
Applying str.lower() function to (drive unit , Volume,FuelType) Columns
Splitting Car-info columns into 3 column (brand , model , year) then dropping brand and model columns
![image](https://github.com/Ahmed-Elsharkaw/Car-price-prediction/assets/113799131/a2e873bc-32af-4968-971e-90eef2078d81)
Adding (mean Price) price column by getting the  mean price for every unique value in car-info column from price(USD) column 
Storing values in dictionary(Every Unique value in car info ,meanPrice)
Storing mean price of each brand in a dictionary (brand,MeanPrice)
![image](https://github.com/Ahmed-Elsharkaw/Car-price-prediction/assets/113799131/d43dcec2-94fb-4318-99e3-d68597be1367)


testing preprocessing :
Mean Price Column is filled according to 4 scenarios 
1 – our dictionary has the car-info key then we simply  return the mean price
2 – our dictionary has same brand and model in car-info key but year is different then we simply return the mean price of closest year difference of same model 
3 – our dictionary has same brand in car info key but model is different we simply return the mean price of brand
4-our dictionary has doesn’t contain Brand then we simply return mean price of Price(usd) column  
![image](https://github.com/Ahmed-Elsharkaw/Car-price-prediction/assets/113799131/44ab530b-e246-4317-82d9-51cf44750514)

categorical encoding:
We apply hot one encoding to 4 columns (condition , fuel type, transmission, drive unit)
![image](https://github.com/Ahmed-Elsharkaw/Car-price-prediction/assets/113799131/e306ff3a-fb24-4f80-8aeb-34840c25350c)


scalling:
We apply feature scaling to 3 coulmns (mean price – volume - mileage) using sklearn standard scaling 
![image](https://github.com/Ahmed-Elsharkaw/Car-price-prediction/assets/113799131/e9e620cc-02a1-4829-a68d-25a24bf388cc)


![Screenshot]()

