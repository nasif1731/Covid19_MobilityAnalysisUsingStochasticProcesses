import os
from flask import Flask, render_template, request, render_template_string, url_for, send_file
from modules.preprocess import load_csv, get_mobility_states
from modules.visuals import plot_steady_pie, plot_state_timeline, plot_mm1_summary
from modules.markov_model import (
    build_transition_matrix,
    compute_steady_state,
    compute_recurrence_times,
    compute_first_passage,
    compute_absorption
)
from modules.hmm_model import forward_algorithm, viterbi_algorithm, compute_hidden_steady_state
from modules.mm1_queue import mm1_metrics
import csv
from fpdf import FPDF

app = Flask(__name__)

# 🔹 Load Global Data on Startup
DATA_PATH = 'data/Global_Mobility_Report.csv'
GLOBAL_DATA = load_csv(DATA_PATH)

# 🔹 Home Route
@app.route('/')
def index():
    return render_template('index.html')


# 🔹 Markov Model Route
@app.route('/markov', methods=['GET', 'POST'])
def markov():
    if request.method == 'POST':
        # Step 1: Get user input
        country = request.form['country']
        year = int(request.form['year'])
        category = request.form['category']

        try:
            sequence = get_mobility_states(GLOBAL_DATA, country, year, category)
        except Exception as e:
            return render_template('markov.html', error=str(e))

        # Step 2: Run Markov Model computations
        matrix, order = build_transition_matrix(sequence)
        steady_raw = compute_steady_state(matrix)
        steady = {order[i]: steady_raw[i] for i in range(len(order))}
        recurrence = compute_recurrence_times(steady_raw, order)
        passage = compute_first_passage(matrix, order)
        absorption = compute_absorption(matrix, order)

        # Step 3: Save charts & timeline
        img_dir = 'static/plots'
        os.makedirs(img_dir, exist_ok=True)
        pie_path = os.path.join(img_dir, 'steady_pie.png')
        line_path = os.path.join(img_dir, 'state_line.png')
        plot_steady_pie(list(steady.values()), list(steady.keys()), pie_path)

        # Step 4: Summary generation
        dominant_state = max(steady, key=steady.get)
        summary = f"In {year}, the most stable mobility behavior in {country} was '{dominant_state}'. " \
                  f"This means people mostly showed '{dominant_state.lower()}' activity in the category '{category}'."

        # Step 5: Save timeline chart + CSV + PDF
        timeline_csv = os.path.join(img_dir, 'timeline.csv')
        timeline_pdf = os.path.join(img_dir, 'timeline_report.pdf')
        plot_state_timeline(sequence, line_path, csv_path=timeline_csv)

        # Export timeline summary as PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Mobility Timeline Report", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 8, txt=summary)
        pdf.image(line_path, x=10, y=pdf.get_y() + 5, w=190)
        pdf.output(timeline_pdf)

        # Step 6: Render result
        html_block = render_template_string("""
            <h3>State Order:</h3>
            <p>{{ order }}</p>

            <h3>Steady State:</h3>
            <ul>{% for k, v in steady.items() %}<li>{{ k }}: {{ v }}</li>{% endfor %}</ul>

            <h3>Recurrence Time:</h3>
            <ul>{% for k, v in recurrence.items() %}<li>{{ k }}: {{ v }}</li>{% endfor %}</ul>

            <h3>First Passage Time:</h3>
            <ul>
            {% for i, row in passage.items() %}
                <li><strong>{{ i }} →</strong>
                    <ul>
                    {% for j, val in row.items() %}
                        <li>{{ j }}: {{ val }}</li>
                    {% endfor %}
                    </ul>
                </li>
            {% endfor %}
            </ul>

            {% if absorption %}
            <h3>Absorption Times:</h3>
            <ul>
            {% for k, v in absorption.items() %}
                <li>{{ k }}: {{ v }}</li>
            {% endfor %}
            </ul>
            {% endif %}
        """, order=order, steady=steady, recurrence=recurrence, passage=passage, absorption=absorption)

        return render_template("result.html",
                               title=f"{country} Mobility Analysis ({year})",
                               subtitle=f"Category: {category}",
                               content=html_block,
                               back_url=url_for('markov'),
                               pie_chart_url='/' + pie_path,
                               line_chart_url='/' + line_path,
                               summary_text=summary)

    return render_template('markov.html')


# 🔹 Hidden Markov Model Route
@app.route('/hmm', methods=['GET', 'POST'])
def hmm():
    default_states = ['Strict Policy', 'Moderate Policy', 'Normal Mobility']
    
    if request.method == 'POST':
        # Step 1: Parse inputs
        obs_str = request.form.get('obs_seq', '')
        obs_seq = [x.strip() for x in obs_str.split(',') if x.strip()]

        # Step 2: Extract start, transition, and emission probabilities
        start_prob = {
            'Strict Policy': float(request.form.get('start_prob[Strict Policy]', 0.5)),
            'Moderate Policy': float(request.form.get('start_prob[Moderate Policy]', 0.3)),
            'Normal Mobility': float(request.form.get('start_prob[Normal Mobility]', 0.2))
        }

        default_trans = {}
        default_emit = {}
        for state in default_states:
            default_trans[state] = {}
            default_emit[state] = {}
            for to_state in default_states:
                default_trans[state][to_state] = float(request.form.get(f'trans[{state}][{to_state}]', 0.33))
            for obs in ['Low Mobility', 'Moderate Mobility', 'High Mobility']:
                default_emit[state][obs] = float(request.form.get(f'emit[{state}][{obs}]', 0.33))

        # Step 3: Run HMM algorithms
        forward_prob = forward_algorithm(obs_seq, default_states, start_prob, default_trans, default_emit)
        viterbi_path = viterbi_algorithm(obs_seq, default_states, start_prob, default_trans, default_emit)
        steady_hidden = compute_hidden_steady_state(default_states, default_trans)

        # Step 4: Save charts
        img_dir = 'static/plots'
        os.makedirs(img_dir, exist_ok=True)
        viterbi_path_img = os.path.join(img_dir, 'viterbi_path.png')
        steady_pie_img = os.path.join(img_dir, 'hidden_steady_pie.png')

        from modules.visuals import plot_viterbi_path, plot_hidden_steady_pie
        plot_viterbi_path(viterbi_path, viterbi_path_img)
        plot_hidden_steady_pie(steady_hidden, steady_pie_img)

        # Store for download use
        app.config["last_viterbi_path"] = viterbi_path
        app.config["last_steady_hidden"] = steady_hidden

        return render_template('hmm.html',
                               forward_prob=round(forward_prob, 6),
                               viterbi_path=viterbi_path,
                               steady_hidden=steady_hidden,
                               selected_obs=obs_seq,
                               viterbi_chart_url='/' + viterbi_path_img,
                               steady_chart_url='/' + steady_pie_img)

    return render_template('hmm.html',
                           forward_prob=None,
                           viterbi_path=[],
                           steady_hidden={},
                           selected_obs=[])


# 🔹 HMM Report Downloads
@app.route('/hmm/download/pdf')
def download_hmm_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("HMM Report")

    pdf.cell(200, 10, txt="COVID-19 Mobility - HMM Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt="Most Likely Policy Path (Viterbi):", ln=True)
    for i, state in enumerate(app.config.get("last_viterbi_path", []), start=1):
        pdf.cell(200, 10, txt=f"Day {i}: {state}", ln=True)

    pdf.ln(5)
    pdf.cell(200, 10, txt="Steady-State Distribution:", ln=True)
    for state, prob in app.config.get("last_steady_hidden", {}).items():
        pdf.cell(200, 10, txt=f"{state}: {prob:.4f}", ln=True)

    if os.path.exists("static/plots/viterbi_path.png"):
        pdf.image("static/plots/viterbi_path.png", x=10, y=pdf.get_y() + 10, w=90)
    if os.path.exists("static/plots/hidden_steady_pie.png"):
        pdf.image("static/plots/hidden_steady_pie.png", x=110, y=pdf.get_y(), w=90)

    pdf.output("static/plots/hmm_report.pdf")
    return send_file("static/plots/hmm_report.pdf", as_attachment=True)


@app.route('/hmm/download/csv')
def download_hmm_csv():
    filepath = "static/plots/hmm_report.csv"
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Day", "Viterbi State"])
        for i, state in enumerate(app.config.get("last_viterbi_path", []), start=1):
            writer.writerow([i, state])

        writer.writerow([])
        writer.writerow(["State", "Steady Probability"])
        for state, prob in app.config.get("last_steady_hidden", {}).items():
            writer.writerow([state, prob])

    return send_file(filepath, as_attachment=True)


# 🔹 Queueing Theory Route (M/M/1)
@app.route('/queue', methods=['GET', 'POST'])
def queue():
    if request.method == 'POST':
        try:
            arrival = float(request.form['arrival'])
            service = float(request.form['service'])
            result = mm1_metrics(arrival, service)

            # Plot queue summary
            plot_path = 'static/plots/mm1_summary.png'
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plot_mm1_summary(result, plot_path)

            # Generate summary text
            rho = result['utilization']
            summary = (
                f"The system utilization is {rho:.2f} "
                f"(server is busy {rho*100:.1f}% of the time). "
                f"On average, {result['L']:.2f} customers are in the system, "
                f"{result['Lq']:.2f} in queue. "
                f"Expected time in system is {result['W']:.2f}, "
                f"of which {result['Wq']:.2f} is spent waiting."
            )

            return render_template('queue.html',
                                   metrics=result,
                                   summary=summary,
                                   chart_url='/' + plot_path)
        except Exception as e:
            return render_template('queue.html', error=str(e))

    return render_template('queue.html')


# 🔹 Run App
if __name__ == '__main__':
    app.run(debug=True)
