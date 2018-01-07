#include "game_utils.hpp"

namespace game{

    double scalar_product(const vec2d& v1, const vec2d& v2)
    {
        return v1.x*v2.x + v1.y*v2.y;
    }

    double vector_product(const vec2d& v1, const vec2d& v2)
    {
        return v1.x*v2.y - v1.y*v2.x;
    }

    double getAngle(const vec2d& v1, const vec2d& v2)
    {
        double scalar_prod = scalar_product(v1,v2);
        double vector_prod = vector_product(v1,v2);

        double len_prod = v1.len() * v2.len();
        if(len_prod < 10e-5) return 0.0;

        scalar_prod /= len_prod;
        vector_prod /= len_prod;

        double alpha = acos(scalar_prod);
        double beta  = asin(vector_prod);

        if (beta < 0.0) alpha *= -1.0;

        return alpha;
    }

    void Controller::update_state(double alpha, double beta, double dt){
        vec2d accel = alpha*unit->max_ds*unit->direction - unit->friction_k*unit->speed.len()*unit->speed;
        unit->direction.rotate(unit->dw*dt);
        unit->position = unit->position + unit->speed * dt;
        unit->speed = unit->speed + unit->accel * dt;
        unit->accel = accel;
        unit->dw = beta*unit->max_dw;
    }

   void  StrategyController::update_state(World& world){
        if(strategy != nullptr){
            Tuple res = strategy->implement(world);
            Controller::update_state(res.first, res.second, world.dt);
        }
   }

    Tuple TargetFollowStrategy::implement(World& world){
        if(controller != nullptr && world.targets.size() > 0)
        {
            MovableUnit* unit = controller->getUnit();

            size_t min_index = 0;
            double min_dist = (world.targets[0]->position - unit->position).len();
            double tmp_dist = 0.0;

            // TODO Check if target id equeals to unit id ( skip this target if so )
            for(size_t i = 1;i < world.targets.size();i++)
            {
                tmp_dist = (world.targets[i]->position - unit->position).len();
                if (tmp_dist < min_dist){
                    min_dist = tmp_dist;
                    min_index = i;
                }
            }

            Unit& target = *world.targets[min_index];
            double target_angle = getAngle(unit->direction, target.position - unit->position);
            double alpha = 0.0, beta = 0.0;

            if (min_dist >= target.radious + unit->radious)
            {
                alpha = 1.0;
                beta  = std::min(1.0, target_angle / 2.0*sqrt(unit->max_dw));
            }

            return Tuple(alpha, beta);
        }

        else{
            return Tuple(0.0, 0.0);
        }
    }

    void World::update_state(double Dt){
        dt = Dt;

        for(size_t i = 0; i < players.size(); i++)
        {
            players[i]->controller.update_state(*this);

            if (players[i]->position.x < 0.0) players[i]->position.x = 0.0;
            if (players[i]->position.y < 0.0) players[i]->position.y = 0.0;
            if (players[i]->position.x > shape.first) players[i]->position.x = shape.first;
            if (players[i]->position.y > shape.second)players[i]->position.y = shape.second;
        }

        for(size_t i = 0; i < targets.size(); i++)
        {
            for(size_t j = 0; j < players.size(); j++)
            {
                if (players[j]->id != targets[i]->id){
                    if((players[j]->position - targets[i]->position).len() <= players[j]->radious + targets[i]->radious){
                        scoreboard[players[j]->id] += 1.0;
                        targets[i]->position.x = std::rand() % (int)(shape.first  + 1);
                        targets[i]->position.y = std::rand() % (int)(shape.second + 1);
                    }
                }
            }
        }
    }

    void World::reset_state(){
        for(size_t i = 0; i < players.size(); i++)
        {        
            players[i]->position.x = std::rand() % (int)(shape.first  + 1);
            players[i]->position.y = std::rand() % (int)(shape.second + 1);        
            scoreboard[players[i]->id] = 0.0;
        }

        for(size_t i = 0; i < targets.size(); i++)
        {
            targets[i]->position.x = std::rand() % (int)(shape.first  + 1);
            targets[i]->position.y = std::rand() % (int)(shape.second + 1);
        } 
    }
          
}