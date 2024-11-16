// include/hello.hpp
// C++ Source File
#ifndef HELLO_HPP  // Include guard to prevent multiple inclusions
#define HELLO_HPP

#include <iostream>
#include <string>

// extern "C" disables name mangling, making the function callable from C
// and other languages that expect C-style linkage.
// Without extern "C", name mangling is enabled,
// and the function is only directly usable from C++.
// C++ function declaration with a default message
extern "C" void print_message(
    const std::string& message = "Hello, from C++!"
) {
  // Print the message
  std::cout << message << std::endl;
}

#endif // HELLO_HPP