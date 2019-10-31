using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class notesMove : MonoBehaviour {
    float speed = -8f;
    HitZone hz;
    Rigidbody2D rb;
    // Use this for initialization
    void Start () {
        hz = GameObject.FindWithTag("Player").GetComponent<HitZone>();
        rb = GetComponent<Rigidbody2D>();
    }
	
	// Update is called once per frame
	void FixedUpdate() {
        //transform.position += new Vector3(speed, 0, 0);
        rb.velocity = new Vector3(speed, 0, 0);
        if (this.gameObject.transform.position.x < -4)
        {
            Destroy(this.gameObject);
            hz.combo = 0;
        }
    }
}
