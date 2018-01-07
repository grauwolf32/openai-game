#include <vector>
#include <iostream>
#include <ctime>

#include "game_utils.hpp"

using namespace game;

int main()
{
    Tuple world_shape(640, 480);
    StrategyController controller1;
    TargetFollowStrategy tf_strategy;

    ControlledUnit player1(vec2d(), vec2d(1.0,0.0), 20, 1, 6, 2, 0.013, controller1);
    player1.controller.bind_strategy(&tf_strategy);

    Unit target1(vec2d(),vec2d(),20, 2);
    Unit target2(vec2d(),vec2d(),20, 3);

    std::vector<ControlledUnit*> players;
    std::vector<Unit*> targets;

    players.push_back(&player1);
    targets.push_back(&target1);
    targets.push_back(&target2);

    World world(players, targets, world_shape, 1.0/10.0);

    size_t n = 0;
    size_t total = 0;
    size_t n_iter = 4000;

    #ifdef __DEBUG__
    for(size_t i = 0; i < world.players.size();i++)
    {
        std::cout << "--------------------" << std::endl;
        std::cout << "Player id:" << world.players[i]->id << std::endl;
        std::cout << "Position: ("<< world.players[i]->position.x <<","<< world.players[i]->position.y <<")"<< std::endl;
        std::cout << "Acceleration: ("<< world.players[i]->accel.x <<","<< world.players[i]->accel.y <<")"<< std::endl;
        std::cout << "Speed: ("<< world.players[i]->speed.x <<","<< world.players[i]->speed.y <<")"<< std::endl;
        std::cout << "Direction: ("<< world.players[i]->direction.x <<","<< world.players[i]->direction.y <<")"<< std::endl;
        std::cout << "dw: "<<  world.players[i]->dw << std::endl;
        std::cout << "--------------------" << std::endl << std::endl;
    }
    #endif
    clock_t time_1 = clock(); 
    for(size_t i = 0; i < n_iter;i++)
    {
        //std::cout << i << std::endl;

        while(n < 2000)
        {
            world.update_state(1.0/10.0);
            #ifdef __DEBUG__
            for(size_t i = 0; i < world.players.size();i++)
            {
                std::cout << "--------------------" << std::endl;
                std::cout << "Player id:" << world.players[i]->id << std::endl;
                std::cout << "Position: ("<< world.players[i]->position.x <<","<< world.players[i]->position.y <<")"<< std::endl;
                std::cout << "Acceleration: ("<< world.players[i]->accel.x <<","<< world.players[i]->accel.y <<")"<< std::endl;
                std::cout << "Speed: ("<< world.players[i]->speed.x <<","<< world.players[i]->speed.y <<")"<< std::endl;
                std::cout << "Direction: ("<< world.players[i]->direction.x <<","<< world.players[i]->direction.y <<")"<< std::endl;
                std::cout << "dw: "<<  world.players[i]->dw << std::endl;
                std::cout << "--------------------" << std::endl;
            }

            for(size_t i = 0; i < world.targets.size();i++)
            {
                std::cout << "Target id:" << world.targets[i]->id << std::endl;
                std::cout << "Position: ("<< world.targets[i]->position.x <<","<< world.targets[i]->position.y <<")"<< std::endl;
                std::cout << "--------------------" << std::endl << std::endl;
            }

            #endif
            n++;
        }

        world.reset_state();
        total += n;
        n = 0;
    }
    clock_t time_2 = clock();
    double seconds = (double)(time_2 - time_1) / CLOCKS_PER_SEC;
    std::cout << "Time: " << seconds << std::endl;

    double t = total;
    double n_i = n_iter;

    double avg_steps = t / seconds; 
    double avg_iter = n_i / seconds;

    std::cout.precision(8);
    std::cout << "Avg steps per second: " << avg_steps << std::endl;
    std::cout << "Avg iters per second: " << avg_iter << std::endl;

    return 0;
}
