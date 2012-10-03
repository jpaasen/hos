#pragma once

// GLUT idle callback function
void idle();

// GLUT keybord callback function
void keyboard(unsigned char key, int x, int y);

// GLUT mouse clicked callback function
void mouse(int button, int state, int x, int y);

// GLUT mouse moved callback function
void motion(int x, int y);

// GLUT arrow keys callback function 
void arrow_keys( int a_keys, int x, int y );

// GLUT window reshape callback function
void reshape(int x, int y);

// GLUT re-display callback function
void display();

// GLUT atExit clean up function
void cleanUpAtExit();