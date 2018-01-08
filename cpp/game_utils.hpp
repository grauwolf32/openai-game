
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>
#include <algorithm>
#include <map>

#pragma once

//#define __DEBUG__

namespace game{
    typedef std::pair<double, double> Tuple;
    typedef std::map<size_t, double> Scoreboard;
    
    class World;

    class vec2d{
        public:
            vec2d(double X, double Y):x(X),y(Y) {};
            vec2d():x(0.0), y(0.0) {};

            double x;
            double y;

            double len() const { return sqrt(x*x + y*y); }
            vec2d  norm(){ 
                double length = this->len();
                if (length > 10e-4){
                    return vec2d(x/length,y/length);
                }
                else {
                    return vec2d(0.0,0.0);
                }
            } 

            vec2d operator+(const vec2d& rhs){
                return vec2d(x + rhs.x, y + rhs.y);
            }

            vec2d operator-(const vec2d& rhs){
                return vec2d(x - rhs.x, y - rhs.y);
            }

            friend vec2d operator*(const vec2d& v, const double alpha){
                return vec2d(v.x*alpha, v.y*alpha);
            }

            friend vec2d operator*(double alpha, const vec2d& v){
                return vec2d(v.x*alpha, v.y*alpha);
            }

            void rotate(double alpha)
            {
                double X = x;
                double Y = y;

                x = X*cos(alpha) - Y*sin(alpha);
                y = X*sin(alpha) + Y*cos(alpha);
            }
    };

    double scalar_product(const vec2d& v1, const vec2d& v2);
    double vector_product(const vec2d& v1, const vec2d& v2);
    double getAngle(const vec2d& v1, const vec2d& v2);

    class Unit{
        public:
            Unit(vec2d p, vec2d d, double r, size_t Id):position(p),direction(d),radious(r),id(Id){};
            Unit():position(vec2d()),direction(vec2d()),radious(0.0),id(0){};
            
            vec2d position;
            vec2d direction;
            
            double radious;
            size_t id;
    };                

    class MovableUnit: public Unit{
        public:
            MovableUnit(vec2d p, vec2d d, double r, size_t Id, double Max_ds, double Max_dw, double Friction_k):Unit(p,d,r,Id)
            {
                max_ds = Max_ds;
                max_dw = Max_dw;
                friction_k = Friction_k;

                speed = vec2d();
                accel = vec2d();
                dw = 0.0;
            }
            
            double max_ds;
            double max_dw;
            double friction_k;

            vec2d speed;
            vec2d accel;
            double dw;
    };

    class Controller{
        public:
            
            void bind_unit(MovableUnit* u){
                unit = u;
            }

            void update_state(double alpha, double beta, double dt);
            MovableUnit* getUnit() { return unit;}

        protected:
            MovableUnit* unit;
    };

    template<class T>
    class Strategy{
        public:
            Strategy(){
                controller = (Controller*)nullptr;
            }
            void bind_controller(Controller* c){
                controller = c;
            }

            virtual T implement(World& world) = 0;

        protected:
            Controller* controller;
    };

    class StrategyController:public Controller{
        public:
            StrategyController(){
                strategy = (Strategy<Tuple>*)nullptr;
            }
            void bind_strategy(Strategy<Tuple>* s){
                strategy = s;
                s->bind_controller(this);
            }
            void update_state(World& world);

        private:
            Strategy<Tuple>* strategy;
    };

    class ControlledUnit:public MovableUnit{
        public:
            ControlledUnit(vec2d p, vec2d d, double r, size_t Id, double Max_ds, double Max_dw, double Friction_k, StrategyController c):MovableUnit(p,d,r,Id,Max_ds,Max_dw,Friction_k), controller(c) {
                controller.bind_unit(this);
            }
            StrategyController controller;
    };

    class TargetFollowStrategy:public Strategy<Tuple>{
        public:
            Tuple implement(World& world);
    };

    class World{
        public:
            World(std::vector<ControlledUnit*>& Players, std::vector<Unit*>& Targets, Tuple Shape, double dt_){
                players = Players;
                targets = Targets;
                shape = Shape;
                dt = dt_;

                reset_state();
            }

            std::vector<ControlledUnit*> players;
            std::vector<Unit*> targets;

            Scoreboard scoreboard;
            Tuple shape;
            double dt;

            void update_state(double Dt);
            void reset_state();
    };
}