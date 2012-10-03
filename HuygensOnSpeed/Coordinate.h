/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

/**
* Template for 3D coordinate 
* Defines a set of functions for operating on 3D coordinates/vectors
**/
template<typename T>
struct Coordinate {

	T x;
	T y;
	T z;

	Coordinate() 
	{
		x = 0; y = 0; z = 0;
	}

	Coordinate(T x, T y, T z) : x(x), y(y), z(z) {}

	/** Returns the length of this vector **/
	T length() 
	{
		return sqrt(x*x + y*y + z*z);
	}

	/** Normalize this vector **/
	void normalize() 
	{
		T vecLength = length();
		x /= vecLength;
		y /= vecLength;
		z /= vecLength;
	}

	/** multiply this vector with a scalar **/
	void mul(T a)
	{
		x *= a; y *= a; z *= a;
	}

	/** add a scalar to this vector **/
	void add(T a)
	{
		x += a; y += a; z += a;
	}

	/** Calcs the product of all elements **/
	T reduceMul()
	{
		return x * y * z;
	}

	/** Calcs the sum of all elements **/
	T reduceAdd()
	{
		return x + y + z;
	}

	/** Calcs the linear interpolation between a.x and b.x with respect to t.x (the same for y and z) **/
	static Coordinate<T> elemlerp(Coordinate<T> a, Coordinate<T> b, Coordinate<T> t)
	{
		return Coordinate<T>(t.x*a.x + (1-t.x)*b.x, t.y*a.y + (1-t.y)*b.y, t.z*a.z + (1-t.z)*b.z);
	}

	/** Subracts per element b from a **/
	static Coordinate<T> subtract(Coordinate<T> a, Coordinate<T> b)
	{
		return Coordinate<T>(a.x - b.x, a.y - b.y, a.z - b.z);
	}

	/** Add per element a and b **/
	static Coordinate<T> add(Coordinate<T> a, Coordinate<T> b)
	{
		return Coordinate<T>(a.x + b.x, a.y + b.y, a.z + b.z);
	}

	/** Multiply per elem a and b **/
	static Coordinate<T> mul(Coordinate<T> a, Coordinate<T> b)
	{
		return Coordinate<T>(a.x * b.x, a.y * b.y, a.z * b.z);
	}

	/** Return a vector containing zeros **/
	static Coordinate<T> zeros()
	{
		return Coordinate<T>(static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0));
	}
};