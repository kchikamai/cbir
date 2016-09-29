#ifndef MYDATATYPE_HPP
#define MYDATATYPE_HPP

#include <iostream>
#include <regex> // for round.. replace with correct maths library

using namespace std;

class cMyDataType // Abstraction container for node-node interchange
{
	public:
		inline cMyDataType():nRows(0),nCols(0),dtype('i'),data(0){}
		inline cMyDataType(int nr, int nc, char dt, double * d):nRows(nr),nCols(nc),dtype(dt),data(d){}
		cMyDataType operator+(const cMyDataType&); // pointwise vector addition
		cMyDataType operator+(const double);      // scalar addition operation
		cMyDataType operator-(const cMyDataType&);
		cMyDataType operator-(const double);
		cMyDataType operator*(const cMyDataType&);
		cMyDataType operator*(const double);
		cMyDataType operator/(const cMyDataType&);
		cMyDataType operator/(const double);
		cMyDataType operator%(const cMyDataType&);
		cMyDataType operator%(const double);
		cMyDataType deepcopy(); // deep copy
		
	// Data: yes, it's public
		int nRows, nCols; // Number of columns and rows of data (set one of them to 1 for unidimensional vector
		char dtype; // active union data element: 'i'=idata|'d'=ddata
		double *data; // can point to either int or double
};
#endif
