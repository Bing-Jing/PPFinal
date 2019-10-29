using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HitZone : MonoBehaviour {

    public GameObject perfecttx;
    public GameObject goodtx;
    public GameObject badtx;
    public GameObject perfectEx;
    public GameObject goodEx;
    public float blueHitVal = 0;
    public float redHitVal = 0;
    public string status = "";

    public int combo = 0;
	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {

	}
    void judge(GameObject collision) {

        if (Mathf.Abs(collision.transform.position.x - this.gameObject.transform.position.x) > 0.6f)
        {
            //bad
            status = "bad";
            Destroy(Instantiate(badtx, transform.position+new Vector3(0,1.5f,0), transform.rotation),0.2f);
            combo = 0;
        }
        else if (Mathf.Abs(collision.transform.position.x
        - this.gameObject.transform.position.x) < 0.2f)
        {
            //perfect
            status = "perfect";
            Destroy(Instantiate(perfecttx, transform.position + new Vector3(0, 1.5f, 0), transform.rotation), 0.2f);
            Destroy(Instantiate(perfectEx, transform.position, transform.rotation), 0.2f);
            combo += 1;
        }
        else
        {
            //good
            status = "good";
            Destroy(Instantiate(goodtx, transform.position + new Vector3(0, 1.5f, 0), transform.rotation), 0.2f);
            Destroy(Instantiate(goodEx, transform.position, transform.rotation), 0.2f);
            combo += 1;
        }
    }
    private void OnTriggerStay2D(Collider2D collision)
    {
        if (collision.gameObject.tag.Contains("Notes") && Mathf.Abs(collision.transform.position.x - this.gameObject.transform.position.x) < 1f)
        {
            //if (Input.GetAxis("redHit") > 0f)
            if (redHitVal > 0f)
            {
                if (collision.gameObject.tag == "redNotes")
                {
                    judge(collision.gameObject);
                    Destroy(collision.gameObject);
                }
            }
            //else if (Input.GetAxis("blueHit") > 0f)
            else if (blueHitVal > 0f)
            {
                if (collision.gameObject.tag == "blueNotes")
                {
                    judge(collision.gameObject);
                    Destroy(collision.gameObject);
                }
            }
        }
    }
    private void OnTriggerExit2D(Collider2D collision)
    {
        if (collision.gameObject.tag.Contains("Notes"))
        {
            status = "bad";
            Destroy(collision.gameObject);
            combo = 0;
        }
    }
}
