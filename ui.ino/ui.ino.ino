/*
This code is used to run the  CTR UI
*/

//float alpha = 0.9;
//float tau = 10000.0;  //time constant
float alpha = 0.2;   //sensitivity to change (closer to 1, greater filtering)
//unsigned long lastTime = 0;
float prevValues[4] = {0.0, 0.0, 0.0, 0.0};
float filteredPrev[4] = {0.0, 0.0, 0.0, 0.0};
float publishedPrev[4] = {0.0, 0.0, 0.0, 0.0};
float publishThreshold = 0.01;  //amount of change in signal to actually send new value
//float prevFilteredValues[4] = {0.0, 0.0, 0.0, 0.0};
// float sumValues[4] = {0.0, 0.0, 0.0, 0.0};
float filteredValues[4] = {0.0, 0.0, 0.0, 0.0};
int analogIdx[4] = {A0, A1, A2, A3};
int pinLayout[4] = {0, 1, 2, 3}; // pin layout: yaw, pitch, roll, translate
// int uLim[4] = {840.0, 702.0, 1023.0, 1023.0};
// int lLim[4] = {150.0,380.0,0.0,0.0};
int uLim[4] = {950.0, 700.0, 1023.0, 1023.0};
int lLim[4] = {150.0,350.0,0.0,0.0};
// int uLim[4] = {1023.0, 1023.0, 1023.0, 1023.0};
// int lLim[4] = {0.0,0.0,0.0,0.0};
int ledPin = 13;      // select the pin for the LED
int sensorValue = 0;  // variable to store the value coming from the sensor
int numPots = 4;
// int N = 20; //number of averaged values

void setup() {
  // declare the ledPin as an OUTPUT:
  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);
}

void loop() {
  // for each potentiometer (DOF) on controller
  for (int i = 0; i < numPots; i++) {
    // get raw value from signal
    float rawValue = analogRead(analogIdx[i]);

    // normalize to [-1,1]
    float norm = (uLim[i] - rawValue - ((uLim[i]-lLim[i])/2.0)) / ((uLim[i]-lLim[i])/2.0);
    
    // hard boundaries 
    if (norm > 1.0) norm = 1.0;
    if (norm < -1.0) norm = -1.0;

    // low-pass filter
    filteredValues[i] = filteredPrev[i] * (1 - alpha) + norm * alpha;

    // deadband threshold (if value doesn't change by threshold amount, don't update value & set current value to last published value)
    if (abs(filteredValues[i] - publishedPrev[i]) < publishThreshold) {
        filteredValues[i] = publishedPrev[i];
    }

    // update state if above deadband threshold
    filteredPrev[i] = filteredValues[i];
    if (abs(filteredValues[i] - publishedPrev[i]) >= publishThreshold) {
        publishedPrev[i] = filteredValues[i];
    }

    // print with 3 sig figs
    Serial.print(publishedPrev[i], 3);

    // separate each chunk of output with tab
    if (i < numPots - 1) {
        Serial.print("\t");
    }
}
Serial.println();
}


  // // //calculate time step
  // // unsigned long currentTime = millis();
  // // float dt = (currentTime-lastTime) / 1000.0; //dt in seconds
  // // lastTime = currentTime;

  // float rawValue = 0.0;
  // int n = 0;
  // while (n < N) {
  //   ++n;
  //   for (int i = 0; i < numPots; i++) {
  //     rawValue = analogRead(analogIdx[i]);
  //     filteredValues[i] = rawValue * (1-alpha) + prevValues[i]*alpha; //infinite impulse response filter
  //     prevValues[i] = rawValue;
  //     sumValues[i] = sumValues[i] + filteredValues[i];
  //     if (n==N) {
  //       filteredValues[i] = float(sumValues[i])/float(N); //averaging filter
  //       sumValues[i] = 0;
  //     }
  //   }
  // }
  // // for (int i = 0; i < numPots; i++){
  // //   rawValue = analogRead(analogIdx[i]);
  // //   float alpha = 1 - (dt / (tau + dt));
  // //   filteredValues[i] = rawValue * (1 - alpha) + prevValues[i]*alpha;
  // //   prevValues[i] = rawValue;
  // //   prevFilteredValues[i] = filteredValues[i];  //use this to avoid jumps from bad pot content

  // // }

  // // read the value from the sensor:


  // float printVal;
  // for (int i = 0; i < numPots; i++){
  //   float currValue = float(filteredValues[i]);
  //   if (pinLayout[i] < 4){
  //     printVal = (uLim[i] - filteredValues[i] - ((uLim[i]-lLim[i])/2))/((uLim[i]-lLim[i])/2);
  //     // printVal = filteredValues[i];
  //     if(printVal > 1.0){
  //       printVal = 1.0;
  //     }else if(printVal < -1.0){
  //       printVal = -1.0;
  //     }
  //     //if ( fabs(printVal - lastValues[i]) > 0.01) {
  //     lastValues[i] = printVal;
  //     Serial.print(printVal,3);
  //     //} 
  //   }
  //   if (i<numPots-1){
  //     Serial.print("\t"); // change this line to change separating value 
  //   }
  // }
  // Serial.println();
// }

  // float printVal;
//   for (int i = 0; i < numPots; i++){
//     float currValue = float(filteredValues[i]);
//     if (pinLayout[i] < 4){
//       printVal = (uLim[i] - filteredValues[i] - ((uLim[i]-lLim[i])/2.0))/((uLim[i]-lLim[i])/2.0);
//       // printVal = filteredValues[i];
//       //Serial.print(printVal,3);
//     }else{
//       printVal = (uLim[i] - filteredValues[i])/(uLim[i]-lLim[i]);
//       // printVal = filteredValues[i];
//       //Serial.print(printVal,3);
//     }
//     if ((printVal > 1.0) || (printVal < -1.0)) {
//       printVal = prevFilteredValues[i];
//     } else {
//       prevFilteredValues[i] = printVal;
//     }
//     Serial.print(printVal,3);
//     if (i<numPots-1){
//       // Serial.print(", ");
//       Serial.print('\t');
//     }
//   }
//   Serial.println();
// }