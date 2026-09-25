/*=========================================================================
  Program:   Teensy UI
  Language:  C++/C

  Jason Shrand, Dominick Ropella, Vanderbilt University 2020. All rights reserved.
  DO NOT REDISTRIBUTE WITHOUT PERMISSION

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE. See the above notices for more information.

=========================================================================*/
#include "AverageFilter.h"


AverageFilter::AverageFilter(std::size_t size)
{
	// Initialize array, and set all elements to zero
	array = new double[size]{};
	array_length = size;
	reset();
}

// Pushes a new double into the moving average
void AverageFilter::pushValue(double value)
{
	double oldest_value = array[oldest_index]; // Get the oldest value before we overwrite it

    array[oldest_index] = value; // Replace the value at the oldest index with the new value

    if(num_values < array_length)
    {
        num_values = num_values + 1; // A new item has been added, and we haven't hit capacity yet
    }

    // Compute the new overall sum and moving average
    moving_sum = moving_sum - oldest_value + value;
    moving_average = moving_sum / num_values;

    // Update the oldest index to the next-oldest item
    oldest_index = oldest_index + 1;
    if(oldest_index >= array_length)
    {
    	oldest_index = 0;
    }
}

// Clears the average filter, and sets the average back to zero
void AverageFilter::reset()
{
    for(unsigned int i = 0; i < array_length; i++)
	{
		array[i] = 0;
	}
	num_values = 0;
	oldest_index = 0;
	moving_sum = 0;
	moving_average = 0;
}

// Returns the current moving average
double AverageFilter::getAverage()
{
	return moving_average;
}
