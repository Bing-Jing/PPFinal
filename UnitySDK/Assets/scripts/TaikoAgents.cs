using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class TaikoAgents : Agent {

    public HitZone hz;
    int oldcombo = 0;
    public override void AgentAction(float[] vectorAction, string textAction)
    {

        int dbtn = (int)vectorAction[0];
        //int fbtn = (int)vectorAction[1];
        //int stall = (int)vectorAction[2];
        //if(stall == 1) {
        //    hz.redHitVal = 0;
        //    hz.blueHitVal = 0;
        //}
        if (dbtn == 0)
        {
            hz.blueHitVal = 0;
            hz.redHitVal = 0;
        }
        else if (dbtn == 1)
        {
            hz.blueHitVal = 1;
            hz.redHitVal = 0;
        }
        else
        {
            hz.blueHitVal = 0;
            hz.redHitVal = 1;

        }
        //if(hz.combo == 0)
        //{
        //    SetReward(-1);
        //}
        if ((oldcombo != 0 && hz.combo == 0) || (hz.status == "bad" && oldcombo == 0))
        {
            hz.status = "None";
            SetReward(-1);
            Done();


        }
        if(hz.combo > oldcombo)
        {
            SetReward(1);
        }
        oldcombo = hz.combo;
        if (hz.combo > 100) {
            oldcombo = 0;
            hz.combo = 0;
            Done();
            SetReward(1);
        }
        if (GetReward() > 0)
        print("combo="+hz.combo+","+oldcombo+","+"dbtn="+dbtn+","+ GetReward());
        //print(dbtn);

    }
    public override void AgentReset()
    {
        print("Reset");
        SetReward(0);
        Destroy(GameObject.FindWithTag("redNotes"));
        Destroy(GameObject.FindWithTag("blueNotes"));

    }
    public override void CollectObservations()
    {

    }

}
