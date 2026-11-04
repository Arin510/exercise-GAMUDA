import './PersonalInfo.css'

function PersonalInfo() {
    const name= "Nur Ain"
    const language ="Javascript"
    const hobbies = ["sleeping","eating","coding"]
    const now = new Date(). toLocaleString()

    return (
        <div class="personal-info">
            <h1>{name}</h1>
            <p>My favourite programming language is {language}</p>

            <h3>My Hobbies:</h3>
            <ul>
                {hobbies.map((hobby, index)=> (
                    <li key={index}>{hobby}</li>
                ))}
            </ul>

            <p>Current date and time: {now}</p>

            <img
            src="https://tse3.mm.bing.net/th/id/OIP.OHd_brVBmHLfyU_i04tSiQHaE8?r=0&rs=1&pid=ImgDetMain&o=7&rm=3"
            alt= "Random photo"
            />
        </div>
    )
}

export default PersonalInfo