// using standard exceptions
#include <iostream>
#include <exception>
#include <stdexcept>

using namespace std;

class myexception: public exception
{
  virtual const char* what() const throw()
  {
    return "My exception happened";
  }
} myex;

int main () {
  try
  {
    int a = 1/(1-1);
  }
  catch(logic_error e)
  {
      cout << "logic catch" << endl;
  }
  catch(runtime_error e)
  {
      cout << "runtime catch" << endl;
  }
  return 0;
}

k = (a == b) ? s : d;


(*s).method()
s->method()