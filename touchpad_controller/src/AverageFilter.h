/*=========================================================================

  Program:   Teensy UI
  Language:  C++/C

ason Shrand, Vanderbilt University 2020. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/

// This utility class creates a moving average filter of N doubles

#ifndef AVERAGE_FILTER_H
#define AVERAGE_FILTER_H

#include <cstddef>

class AverageFilter{

    public:

    	// Creates a new moving average filter
        AverageFilter(std::size_t size);

        // Pushes a new double into the moving average
        void pushValue(double value);

        // Clears the average filter, and sets the average back to zero
        void reset();

        // Returns the current moving average
        double getAverage();

    private:

    	std::size_t array_length;
    	unsigned int num_values; // The number of values registered so far
    	unsigned int oldest_index; // The index of the oldest element. This is the one that will be replaced next time a value is pushed
    	double *array;
    	double moving_sum;     // The total sum of all values stored so far
    	double moving_average; // The moving average

};


#endif // AVERAGE_FILTER_H