"""Run against a local gallery server: python tests/browser/smoke.py URL."""
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / 'site/js/gallery.js').read_text()
probe = '''
window.probe = {
  get session() { return current.simulation; },
  get input() { return input; },
  screen(point) {
    const [x, y] = renderer.projection.toScreen(point);
    const rect = canvas.getBoundingClientRect();
    return [rect.left + x, rect.top + y];
  },
  draw() { renderer.draw(current.simulation.scene(), input.overlay()); },
  fixed() {
    current.simulation.state = {positions: [[1, 0], [3, 4]], target: 0, active: false, steps: 0};
    current.simulation.previous = current.simulation.state;
    this.draw();
  },
  overlap() {
    current.simulation.state = {...current.simulation.state, positions: [[2, 2], [2, 2]]};
    current.simulation.previous = current.simulation.state;
    this.draw();
  },
  switchDemo() { return activate(demos[0]); },
};
'''

with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={'width': 1100, 'height': 950}, reduced_motion='reduce')
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.route('**/js/gallery.js', lambda route: route.fulfill(
        status=200, content_type='text/javascript', body=source + probe))
    page.goto(sys.argv[1])
    page.wait_for_function('window.probe')
    assert page.locator('#play').inner_text() == 'Play'
    page.evaluate('probe.fixed()')
    start = page.evaluate('probe.screen([1, 0])')
    end = page.evaluate('probe.screen([1.35, 1.25])')
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end)
    assert page.evaluate('probe.session.held') == 0
    page.evaluate("document.querySelector('canvas').dispatchEvent(new PointerEvent('pointerup', {pointerId: 77}))")
    assert page.evaluate('probe.session.held') == 0
    page.mouse.up()
    assert page.evaluate('probe.session.state.positions[0]') == [1, 1]
    page.evaluate('probe.session.playing=true;probe.session.advance(.65);probe.session.playing=false')
    assert page.evaluate('probe.session.state.positions[0]') == [1, 1]
    assert page.evaluate('probe.session.held') is None
    page.locator('#reset').click()
    page.evaluate('probe.overlap()')
    overlap = page.evaluate('probe.screen([2, 2])')
    page.mouse.click(*overlap)
    assert page.evaluate('probe.input.selected') == 1
    page.mouse.click(*overlap)
    assert page.evaluate('probe.input.selected') == 0
    page.locator('#reset').click()
    page.evaluate('probe.session.state.positions[0]=[0,0];probe.draw()')
    edge = page.evaluate('probe.screen([0,0])')
    page.mouse.click(2, edge[1])
    assert page.evaluate('probe.session.held') is None
    page.mouse.click(*edge, button='right')
    assert page.evaluate('probe.session.held') is None
    page.mouse.move(*edge)
    page.mouse.down()
    page.mouse.move(500, 30)
    page.mouse.up()
    assert page.evaluate('probe.input.pointerId') is None
    page.locator('#reset').click()
    page.evaluate('probe.fixed()')
    page.locator('canvas').focus()
    page.keyboard.press('Enter')
    page.keyboard.press('ArrowUp')
    assert page.evaluate('probe.session.state.positions[0]') == [1, 1]
    page.evaluate('probe.switchDemo()')
    assert page.evaluate('probe.session.state.steps') == 0
    assert page.evaluate('probe.input.pointerId') is None
    page.evaluate('probe.fixed()')
    page.locator('#play').click()
    start = page.evaluate('probe.screen([1,0])')
    end = page.evaluate('probe.screen([0.35,3.25])')
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end, steps=5)
    page.wait_for_timeout(720)
    assert page.evaluate('probe.session.state.steps') >= 1
    assert page.evaluate('probe.session.state.positions[0]') == [0, 3]
    assert page.evaluate('probe.input.pointerId') is not None
    page.mouse.up()
    assert page.evaluate('probe.session.state.positions[0]') == [0, 3]
    assert page.evaluate('probe.input.pointerId') is None
    for width in [1440, 768, 390]:
        page.set_viewport_size({'width': width, 'height': 950})
        page.wait_for_timeout(50)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors, errors
    browser.close()
