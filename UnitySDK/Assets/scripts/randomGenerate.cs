using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class randomGenerate : MonoBehaviour {
    public GameObject[] notes;
    // Use this for initialization
    float calltime = 1;
    float maxTime = 1f;
    float minTime = 0.5f;
    private float time = 0;
    private float spawnTime = 0;

    void Start () {
        //InvokeRepeating("generateObj", 0.5f, calltime);
        SetRandomTime();

    }
	
	// Update is called once per frame
	void Update () {
        time += Time.deltaTime;
        if(time >= spawnTime)
        {
            generateObj();
            SetRandomTime();
        }
    }
    void generateObj() {
        time = 0;
        int Random_Objects = Random.Range(0, notes.Length);
        GameObject tmp = Instantiate(notes[Random_Objects], transform.position, transform.rotation);
        
        //tmp.transform.parent = this.gameObject.transform;
    }
    void SetRandomTime()
    {
        spawnTime = Random.Range(minTime, maxTime);
    }

}
