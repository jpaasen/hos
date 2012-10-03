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
* Template for 2D dimension
**/
template<typename T>
struct Dimension {
	T w;
	T h;

	Dimension(T w, T h) : w(w), h(h) {}
};