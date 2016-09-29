#include "mydatatype.hpp"

cMyDataType cMyDataType::operator+(const cMyDataType& RHS)
{
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(this->data && RHS.data && (this->nRows==RHS.nRows) && (this->nCols==RHS.nCols)){
		for(int i=0;i<this->nRows*this->nCols;++i)
			c.data[i] = this->data[i]+RHS.data[i];
	} else printf("Invalid array data or dimensions\n");
	return c;
}
cMyDataType cMyDataType::operator+(const double RHS)
{
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(this->data){
		for(int i=0;i<this->nRows*this->nCols;++i)
			c.data[i] = this->data[i]+RHS;
	} else printf("Invalid array data\n");
	return c;
}
cMyDataType cMyDataType::operator-(const cMyDataType& RHS)
{
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(this->data && RHS.data && (this->nRows==RHS.nRows) && (this->nCols==RHS.nCols)){
		for(int i=0;i<this->nRows*this->nCols;++i)
			c.data[i] = this->data[i]-RHS.data[i];
	} else printf("Invalid array data or dimensions\n");
	return c;
}
cMyDataType cMyDataType::operator-(const double RHS)
{
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(this->data){
		for(int i=0;i<this->nRows*this->nCols;++i)
			c.data[i] = this->data[i]-RHS;
	} else printf("Invalid array data\n");
	return c;
}
cMyDataType cMyDataType::operator*(const cMyDataType& RHS)
{
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(this->data && RHS.data && (this->nRows==RHS.nRows) && (this->nCols==RHS.nCols)){
		for(int i=0;i<this->nRows*this->nCols;++i)
			c.data[i] = this->data[i]*RHS.data[i];
	} else printf("Invalid array data or dimensions\n");
	return c;
}
cMyDataType cMyDataType::operator*(const double RHS)
{
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(this->data){
		for(int i=0;i<this->nRows*this->nCols;++i)
			c.data[i] = this->data[i]*RHS;
	} else printf("Invalid array data\n");
	return c;
}
cMyDataType cMyDataType::operator/(const cMyDataType& RHS)
{
	bool divbyzero=false;
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(this->data && RHS.data && (this->nRows==RHS.nRows) && (this->nCols==RHS.nCols)){
		for(int i=0;i<this->nRows*this->nCols;++i){
			if(RHS.data[i]!=0) c.data[i] = this->data[i]/RHS.data[i];
			else divbyzero=true;
		}
	} else printf("Invalid array data or dimensions\n");
	return c;
}
cMyDataType cMyDataType::operator/(const double RHS)
{
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(RHS!=0 && this->data){
		for(int i=0;i<this->nRows*this->nCols;++i)
			c.data[i] = this->data[i]/RHS;
	} else printf("Invalid array data\n");
	return c;
}
cMyDataType cMyDataType::operator%(const cMyDataType& RHS)
{
	// Apply modulus. Truncate double value to its integer equivalent
	bool divbyzero=false;
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(this->data && RHS.data && (this->nRows==RHS.nRows) && (this->nCols==RHS.nCols)){
		for(int i=0;i<this->nRows*this->nCols;++i){
			if(RHS.data[i]!=0) c.data[i] = (int)round(this->data[i])%(int)round(RHS.data[i]);
			else divbyzero=true;
		}
	} else printf("Invalid array data, dimensions or division by zero\n");
	return c;
}
cMyDataType cMyDataType::operator%(const double RHS)
{
	// Apply modulus. Truncate double value to its integer equivalent
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(RHS!=0 && this->data){
		for(int i=0;i<this->nRows*this->nCols;++i)
			c.data[i] = (int)round(this->data[i])%(int)round(RHS);
	} else printf("Invalid array data or division by zero\n");
	return c;
}
cMyDataType cMyDataType::deepcopy()
{
	cMyDataType c{this->nRows,this->nCols,this->dtype,new double[this->nRows*this->nCols]};
	if(this->data){
		for(int i=0;i<this->nRows*this->nCols;++i)
			c.data[i] = this->data[i];
	}
	return c;
}
