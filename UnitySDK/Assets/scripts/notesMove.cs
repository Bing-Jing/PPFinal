using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class notesMove : MonoBehaviour {
    float speed = -8f;
    HitZone hz;
    // Use this for initialization
    void Start () {
        hz = GameObject.FindWithTag("Player").GetComponent<HitZone>();
	}
	
	// Update is called once per frame
	void Update () {
        transform.position += new Vector3(speed*Time.deltaTime, 0, 0);
        if(this.gameObject.transform.position.x < -4)
        {
            Destroy(this.gameObject);
            hz.combo = 0;
        }
    }
}
